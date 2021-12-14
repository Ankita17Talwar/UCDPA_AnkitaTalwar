import numpy as np

matrix1 = np.array([[1, 2, 3],
                    [3, 4, 5],
                    [7, 6, 4]])

matrix2 = np.array([[5, 2, 6],
                    [5, 6, 7],
                    [7, 6, 4]])

matrix3 = np.array([[1, 2, 3],
                    [3, 4, 5],
                    [7, 6, 4]])

matrix4 = np.array([[5, 2, 6],
                    [5, 6, 7]])

matrix5 = np.array([[1, 2, 3],
                    [3, 4, 5]])

matrix6 = np.array([6, 7, 8])


def validate_matrix_mul(m1, m2):
    """
    Validate if input matrices can be multiplied. i.e c1 == r2

    :param m1: Mandatory: Matrix 1
    :param m2: Mandatory : Matrix 2
    :return: valid_flag = 1, 0
    """
    m1_row = m1.shape[0]
    m1_col = m1.shape[1]

    m2_row = m2.shape[0]
    m2_col = m2.shape[1]

    print("Matrix 1:", '\n', m1)
    print('Shape :', '\n', m1.shape)
    print('Rows = ', m1_row)
    print('Columns = ', m1_col)

    print("Matrix 2:", '\n', m2)
    print('Shape :', '\n', m2.shape)
    print('Rows = ', m2_row)
    print('Columns = ', m2_col)

    if m1_col == m2_row:
        valid_flag = 1
    else:
        valid_flag = 0

    return valid_flag


def multiplication(m1, m2):
    """

    :param m1: Mandatory: Matrix 1
    :param m2: Mandatory: Matrix 2
    :return:
    """
    if validate_matrix_mul(m1, m2):
        result = np.dot(m1, m2)
        print("Multiplication Result", '\n', result)
    else:
        print('Matrix not compatible for multiplication')


def add(m1, m2):
    """

    :param m1: Mandatory: Matrix 1
    :param m2: Mandatory: Matrix 2
    :return:
    """
    try:
        add_res = np.add(m1,m2)
        print("Addition :",add_res)
    except ValueError as e:
        print("**************************************************************")
        print(str(e))
        print("Matrix are not compatible for addition")
        print("**************************************************************")


if __name__ == '__main__':

    # 1 Matrix Multiplication
    multiplication(matrix1, matrix2)
    print("**************************************************************")
    multiplication(matrix3, matrix4)
    print("**************************************************************")

    # 2 Addition
    add(matrix3, matrix4)  # operands could not be broadcast together with shapes (3,3) (2,3)
    add(matrix4, matrix3)  # operands could not be broadcast together with shapes (2,3) (3,3)
    add(matrix5, matrix6)  # [[7 9 11] [9 11 13]]
    add(matrix6, matrix5)  # [[7 9 11] [9 11 13]]
    add(matrix1, matrix2)  # [[ 6  4  9] [ 8 10 12] [14 12  8]]

    # Two matrix are compatible when they are equal ,or when one of them is 1 (ex 3 and 4 above)

    # 4 Load Data from text file
    # Load output file  contains ESG score and price data of Stock_ESG script
    ## file = '../dataset/Stock_Data.csv'
    file = '../output_files/out_esg_info/Transformed_data.csv'

    # genfromtext handles missing values
    data_float = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=range(3, 10), dtype=float)

    #data_float = np.delete(data_float, [0,1,2], axis=1)
    print(data_float[0])
    # [7.39100000e+02 2.04175002e+09 1.50905744e+12 3.68000000e+00
    #  6.50000000e+00 3.54000000e+00 1.37100000e+01]

    print(data_float[:,5])
    # [ 3.54 10.64 11.69  2.71  2.13  3.81  0.   20.48  4.06  0.    2.1  16.84
    #  12.43  2.01  3.82  9.86 10.91 18.49  1.13  2.38  2.3  13.37 22.08 10.67
    #   2.25  2.35  7.15 18.1   6.24  1.07 14.52  2.24  4.6   2.36  8.13  9.8
    #   6.63 17.75 10.97 22.56  0.   20.55 15.    2.33  3.02  0.79  8.57 16.32
    #   0.47  1.64 11.11 18.46  8.6  20.69  7.64  0.95  1.56  0.09]

    # mean of ESG score i.e 6th column
    mean_score = np.mean(data_float[:, 5])
    print('Mean ESG Score using mean function  :', mean_score) ## nan if data contains nan

    mean_score = np.nanmean(data_float[:, 5])
    print('Mean ESG Score using nanmean function  :', mean_score)  ## 28.324878048780487 for 50 Stocks

    min_esg_score = np.nanmin(data_float[:, 5])
    print('Min ESG Score : ', min_esg_score)

    max_esg_score = np.nanmax(data_float[:, 5])
    print('Max ESG Score : ', max_esg_score)

    # Mean ESG Score using mean function  : 7.964827586206897
    # Mean ESG Score using nanmean function  : 7.964827586206897
    # Min ESG Score :  0.0
    # Max ESG Score :  22.56

    sum_market_cap = np.sum(data_float[:, 1])
    print('NIFTY 500 total Market cap : ', sum_market_cap)
    # NIFTY 50 total Market cap :  215207509724.0



