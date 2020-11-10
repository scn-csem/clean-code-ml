def add(num1, num2):
    # Add two numbers
    # 
    # Args
    #     num1 (float): a number.
    #     num2 (fload): a second number.
    #
    # Returns:
    #     _sum (float): the sum.
    _sum = num1 + num2
    return _sum


def multiply_by_10(df):
    columns = df.columns
    for col in columns:
        df[col] = df[col] * 10
    return df 
