from __future__ import division
import numpy as np
from scipy.interpolate import griddata

nan = np.NaN


class CleanFill:
    def fill_linear(in_data: object) -> object:
        """
        Fill NaN values in the input array `indata`.
        """

        in_data = CleanFill.ZeroToNaN(in_data)
        # Find the non-NaN indices

        inds = np.nonzero(~np.isnan(in_data))
        # Create an `out_inds` array that contains all of the indices of indata.
        out_inds = np.mgrid[[slice(s) for s in in_data.shape]].reshape(in_data.ndim, -1).T
        # Perform the interpolation of the non-NaN values to all the indices in the array:
        return griddata(inds, in_data[inds], out_inds, method='linear').reshape(in_data.shape)

    def fill_nearest(in_data: object) -> object:
        """
        Fill NaN values in the input array `indata`.
        """

        in_data = CleanFill.ZeroToNaN(in_data)
        # Find the non-NaN indices

        inds = np.nonzero(~np.isnan(in_data))
        # Create an `out_inds` array that contains all of the indices of indata.
        out_inds = np.mgrid[[slice(s) for s in in_data.shape]].reshape(in_data.ndim, -1).T
        # Perform the interpolation of the non-NaN values to all the indices in the array:
        return griddata(inds, in_data[inds], out_inds, method='linear').reshape(in_data.shape)

    """
    Slope One Predictor - Slope One, Weighted Slope One, Bi-polar Slope One
    Reference:
    - D.Lemire, A.Maclachlan, "Slope One Predictors for Online Rating-Based Collaborative Filtering", SDM'05, (2005)
        http://lemire.me/fr/documents/publications/lemiremaclachlan_sdm05.pdf
    - 森口麻里, "協調フィルタリングを用いた映画評価予測プログラムの試作", (2009)
        http://www.tani.cs.chs.nihon-u.ac.jp/g-2009/mori/sotuken/mori_re.pdf
    """

    # Slope One Predictor
    def fill_slope_one(in_data: object) -> object:
        """
        function implement slope one algorithm
        @param rating matrix
        @return predicted rating matrix
        """
        old_data = in_data

        in_data = CleanFill.NaNToZero(in_data)

        user_num = in_data.shape[0]
        item_num = in_data.shape[1]

        # get average deviation
        def get_dev_val(i, j):
            """
            function to get deviation of item i and item j
            @param item pair
            @return deviation value
            """
            dev_val = 0
            users = 0
            for row in range(user_num):
                # if the user evaluated both movie i and movie j
                if ((in_data[row][i] != 0) and (in_data[row][j] != 0)):
                    users += 1
                    dev_val += in_data[row][i] - in_data[row][j]
            # avoid zero division
            if (users == 0):
                return 0
            return dev_val / users

        # get average diviation
        dev = np.zeros((item_num, item_num))
        for i in range(item_num):
            for j in range(item_num):
                if i == j:
                    # to lessen time complexity
                    break
                else:
                    # dev[i][j] = -dev[j][i]
                    dev_temp = get_dev_val(i, j)
                    dev[i][j] = dev_temp
                    dev[j][i] = (-1) * dev_temp
        # get predictive evaluation matrix
        pred_mat = np.zeros((user_num, item_num))
        for u in range(user_num):
            # only get rated item
            eval_row = np.where(in_data[u] != 0)[0]
            for j in range(item_num):
                pred_mat[u][j] = (np.sum(dev[j][eval_row] + in_data[u][eval_row])) / len(eval_row)
        pred_mat = CleanFill.ReplaceWithOld(pred_mat, old_data)
        return pred_mat

    # weighted slope one
    def fill_weighted_slope_one(in_data: object) -> object:
        """
        function implement weighted slope one algorithm
        @param rating matrix
        @return predicted rating matrix
        """

        old_data = in_data

        in_data = CleanFill.NaNToZero(in_data)

        user_num = in_data.shape[0]
        item_num = in_data.shape[1]

        # get average deviation
        def get_dev_val(i, j):
            """
            function to get deviation of item i and item j
            @param item pair
            @return deviation value and evaled user num
            """
            dev_val = 0
            users = 0
            for row in range(user_num):
                # if the user evaluated both movie i and movie j
                if ((in_data[row][i] != 0) and (in_data[row][j] != 0)):
                    users += 1
                    dev_val += in_data[row][i] - in_data[row][j]
            # avoid zero division
            if (users == 0):
                ret = 0
            else:
                ret = dev_val / users
            return ret, users

        # get average diviation
        dev = np.zeros((item_num, item_num))
        # evaled users matrix,(i, j) element represents number of users who evaluated both item i and item j
        evaled_users_mat = np.zeros((item_num, item_num))
        for i in range(item_num):
            for j in range(item_num):
                if i == j:
                    # to lessen time complexity
                    break
                else:
                    dev_temp, users = get_dev_val(i, j)
                    dev[i][j] = dev_temp
                    dev[j][i] = (-1) * dev_temp
                    evaled_users_mat[i][j] = users
                    evaled_users_mat[j][i] = users
        # get predictive evaluation matrix
        pred_mat = np.zeros((user_num, item_num))
        for u in range(user_num):
            eval_row = np.where(in_data[u] != 0)[0]
            for j in range(item_num):
                pred_mat[u][j] = np.sum(
                    (dev[j][eval_row] + in_data[u][eval_row]) * evaled_users_mat[j][eval_row]) / np.sum(
                    evaled_users_mat[j][eval_row])
        pred_mat = CleanFill.ReplaceWithOld(pred_mat, old_data)
        return pred_mat

    # bi-polar slope one
    def fill_bipolar_slope_one(in_data: object) -> object:
        """
        function implement bipolar slope one algorithm
        @param rating matrix
        @return predicted rating matrix
        """

        old_data = in_data

        in_data = CleanFill.NaNToZero(in_data)

        user_num = in_data.shape[0]
        item_num = in_data.shape[1]

        def average_evaluation(eval_mat):
            """
            function to get average evaluation of each user
            @param rating matrix
            @return average rating of each user, array
            """
            ret = np.mean(eval_mat, axis=1)
            items = eval_mat.shape[1]
            for row in range(ret.shape[0]):
                nonzero = np.count_nonzero(eval_mat[row])
                ret[row] *= items / nonzero
            return ret

        ave_eval_lst = average_evaluation(in_data)

        # get average deviation
        def get_dev_val_like(i, j):
            """
            function to get deviation of liked item i and liked item j
            @param item pair
            @return deviation value and evaled user num
            """
            dev_val = 0
            users_like = 0
            for row in range(user_num):
                # if the user evaluated both movie i and movie j
                threshold = ave_eval_lst[row]
                if ((in_data[row][i] > threshold) and (in_data[row][j] > threshold)):
                    users_like += 1
                    dev_val += in_data[row][i] - in_data[row][j]
            # avoid zero division
            if (users_like == 0):
                ret = 0
            else:
                ret = dev_val / users_like
            return ret, users_like

        # get average deviation
        def get_dev_val_dislike(i, j):
            """
            function to get deviation of disliked item i and disliked item j
            @param item pair
            @return deviation value and evaled user num
            """
            dev_val = 0
            users_dislike = 0
            for row in range(user_num):
                # if the user evaluated both movie i and movie j
                threshold = ave_eval_lst[row]
                if ((0 < in_data[row][i] < threshold) and (0 < in_data[row][j] < threshold)):
                    users_dislike += 1
                    dev_val += in_data[row][i] - in_data[row][j]
            # avoid zero division
            if (users_dislike == 0):
                ret = 0
            else:
                ret = dev_val / users_dislike
            return ret, users_dislike

        # get average diviation
        dev_like = np.zeros((item_num, item_num))
        dev_dislike = np.zeros((item_num, item_num))
        evaled_like_users_mat = np.zeros((item_num, item_num))
        evaled_dislike_users_mat = np.zeros((item_num, item_num))

        for i in range(item_num):
            for j in range(item_num):
                if i == j:
                    break
                else:
                    dev_like_temp, users_like = get_dev_val_like(i, j)
                    dev_like[i][j] = dev_like_temp
                    dev_like[j][i] = (-1) * dev_like_temp
                    evaled_like_users_mat[i][j] = users_like
                    evaled_like_users_mat[j][i] = users_like
                    dev_dislike_temp, users_dislike = get_dev_val_dislike(i, j)
                    dev_dislike[i][j] = dev_dislike_temp
                    dev_dislike[j][i] = (-1) * dev_dislike_temp
                    evaled_dislike_users_mat[i][j] = users_dislike
                    evaled_dislike_users_mat[j][i] = users_dislike

        # get predictive evaluation matrix
        pred_mat = np.zeros((user_num, item_num))
        for u in range(user_num):
            eval_like_row = np.where(in_data[u] > ave_eval_lst[u])[0]
            eval_dislike_row = np.where((in_data[u] < ave_eval_lst[u]) & (in_data[u] > 0))[0]
            for j in range(item_num):
                den = np.sum(evaled_like_users_mat[j][eval_like_row]) + np.sum(
                    evaled_dislike_users_mat[j][eval_dislike_row])
                if den != 0:
                    nume = np.sum((dev_like[j][eval_like_row] + in_data[u][eval_like_row]) * evaled_like_users_mat[j][
                        eval_like_row]) + np.sum((dev_dislike[j][eval_dislike_row] + in_data[u][eval_dislike_row]) *
                                                 evaled_dislike_users_mat[j][eval_dislike_row])
                    pred_mat[u][j] = nume / den
        pred_mat = CleanFill.ReplaceWithOld(pred_mat, old_data)
        return pred_mat

    def ZeroToNaN(indata):
        indata = indata.astype('float')
        indata[indata == 0] = nan
        return indata

    def NaNToZero(indata):
        return np.nan_to_num(indata, nan=0.0)

    def ReplaceWithOld(new_data, old_data):
        columns = old_data.shape[0]
        rows = old_data.shape[1]
        for i in range(columns):
            for j in range(rows):
                if not np.isnan(old_data[i][j]):
                    new_data[i][j] = old_data[i][j]
        return new_data




