import numpy as np
import sys

# return -1 if x < 0, 1 if x > 0, random -1 or 1 if x ==0
def sign(x):
    s = np.sign(x)
    tmp = s[s == 0]
    s[s==0] = np.random.choice([-1, 1], tmp.shape)
    return s

def norm(x, keepdims=False):
    '''
    Param: 
        x: matrix of shape (n1, n2, ..., nk) 
        keepdims: Whether keep dims or not
    Return: norm of matrix of shape (n1, n2, ..., n_{k-1})
    '''
    return np.sqrt(np.sum(np.square(x), axis=-1, keepdims=keepdims))

def normed(x):
    '''
    Param: matrix of shape (n1, n2, ..., nk)
    Return: normed matrix
    '''
    return x / (1e-20 + norm(x, keepdims=True))

def inner_product(x1, x2, pair=False):
    if pair:
        return - np.inner(x1, x2)
    else:
        return - np.sum(x1 * x2, axis=-1)

def euclidean2(x1, x2):
    return np.sum(np.square(x1 - x2), axis=-1)

def normed_euclidean2(x1, x2):
    return euclidean2(normed(x1), normed(x2))

def cosine(x1, x2):
    return (1 + inner_product(normed(x1), normed(x2))) / 2

def distance(x1, x2=None, pair=True, dist_type="euclidean2", ifsign=False):
    '''
    Param:
        x2: if x2 is None, distance between x1 and x1 will be returned.
        pair: if True, for i, j, x1_i, x2_j will be calculated
              if False, for i, x1_i, x2_i will be calculated, and it requires the dimension of x1 and x2 is same.
        dist_type: distance type, can be euclidean2, normed_euclidean2, inner_product, cosine
    '''
    if x2 is None:
        x2 = x1
    if ifsign:
        x1 = sign(x1)
        x2 = sign(x2)
    if dist_type == 'inner_product':
        return inner_product(x1, x2, pair)
    if pair:
        x1 = np.expand_dims(x1, 1)
        x2 = np.expand_dims(x2, 0)
    return getattr(sys.modules[__name__], dist_type)(x1, x2)

def get_mAPs(q_output, q_labels, db_output, db_labels, Rs, dist_type):
    dist = distance(q_output, db_output, dist_type=dist_type, pair=True)
    unsorted_ids = np.argpartition(dist, Rs - 1)[:, :Rs]
    APx = []
    for i in range(dist.shape[0]):
        label = q_labels[i, :]
        label[label == 0] = -1
        idx = unsorted_ids[i, :]
        idx = idx[np.argsort(dist[i, :][idx])]
        imatch = np.sum(np.equal(db_labels[idx[0: Rs], :], label), 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, Rs + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
    return np.mean(np.array(APx))

class MAPs:
    def __init__(self, R):
        self.R = R

    def get_mAPs_by_feature(self, database, query, Rs=None, dist_type='inner_product'):
        if Rs is None:
            Rs = self.R
        return get_mAPs(query.output, query.label, database.output, database.label, Rs, dist_type)

    def get_mAPs_after_sign(self, database, query, Rs=None, dist_type='inner_product'):
        if Rs is None:
            Rs = self.R
        q_output = sign(query.output)
        db_output = sign(database.output)
        return get_mAPs(q_output, query.label, db_output, database.label, Rs, dist_type)

    @staticmethod
    def get_precision_recall_by_Hamming_Radius(database, query, radius=2):
        query_output = sign(query.output)
        database_output = sign(database.output)

        bit_n = query_output.shape[1]

        ips = np.dot(query_output, database_output.T)
        ips = (bit_n - ips) / 2
        ids = np.argsort(ips, 1)

        precX = []
        recX = []
        mAPX = []
        query_labels = query.label
        database_labels = database.label

        for i in range(ips.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = np.reshape(np.argwhere(ips[i, :] <= radius), (-1))
            all_num = len(idx)

            if all_num != 0:
                imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
                match_num = np.sum(imatch)
                precX.append(np.float(match_num) / all_num)

                all_sim_num = np.sum(
                    np.sum(database_labels[:, :] == label, 1) > 0)
                recX.append(np.float(match_num) / all_sim_num)

                if radius < 10:
                    ips_trad = np.dot(
                        query.output[i, :], database.output[ids[i, 0:all_num], :].T)
                    ids_trad = np.argsort(-ips_trad, axis=0)
                    db_labels = database_labels[ids[i, 0:all_num], :]

                    rel = match_num
                    imatch = np.sum(db_labels[ids_trad, :] == label, 1) > 0
                    Lx = np.cumsum(imatch)
                    Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                    if rel != 0:
                        mAPX.append(np.sum(Px * imatch) / rel)
                else:
                    mAPX.append(np.float(match_num) / all_num)

            else:
                precX.append(np.float(0.0))
                recX.append(np.float(0.0))
                mAPX.append(np.float(0.0))

        return np.mean(np.array(precX)), np.mean(np.array(recX)), np.mean(np.array(mAPX))

    @staticmethod
    def get_precision_recall_by_Hamming_Radius_All(database, query):
        query_output = sign(query.output)
        database_output = sign(database.output)

        bit_n = query_output.shape[1]

        ips = np.dot(query_output, database_output.T)
        ips = (bit_n - ips) / 2
        precX = np.zeros((ips.shape[0], bit_n + 1))
        recX = np.zeros((ips.shape[0], bit_n + 1))
        mAPX = np.zeros((ips.shape[0], bit_n + 1))

        query_labels = query.label
        database_labels = database.label

        ids = np.argsort(ips, 1)

        for i in range(ips.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1

            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
            all_sim_num = np.sum(imatch)

            counts = np.bincount(ips[i, :].astype(np.int64))

            for r in range(bit_n + 1):
                if r >= len(counts):
                    precX[i, r] = precX[i, r - 1]
                    recX[i, r] = recX[i, r - 1]
                    mAPX[i, r] = mAPX[i, r - 1]
                    continue

                all_num = np.sum(counts[0:r + 1])

                if all_num != 0:
                    match_num = np.sum(imatch[0:all_num])
                    precX[i, r] = np.float(match_num) / all_num
                    recX[i, r] = np.float(match_num) / all_sim_num

                    rel = match_num
                    Lx = np.cumsum(imatch[0:all_num])
                    Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                    if rel != 0:
                        mAPX[i, r] = np.sum(Px * imatch[0:all_num]) / rel
        return np.mean(np.array(precX), 0), np.mean(np.array(recX), 0), np.mean(np.array(mAPX), 0)