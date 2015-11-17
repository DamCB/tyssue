
def _to_3d(df):

    df_3d = np.asarray(df).repeat(3).reshape((df.size, 3))
    return df_3d

