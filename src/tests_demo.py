import unittest
from src.demo import add, multiply_by_10
import pandas as pd
from pandas.testing import assert_frame_equal

# Define a parent test class
class TestDemo(unittest.TestCase):
    def test_1_should_equal_1(self):
        # Here, we assert that 1 == 1
        self.assertEquals(1, 1) 

    def test_add_1_and_1_should_equal_2(self):
        self.assertEquals(2, add(1, 1))

    def test_multiply_by_10_should_give_a_df_with_values_multiplied_by_10(self):
        # Arrange
        df = pd.DataFrame({
            'column_1' : [1, 2, 3]
        })
        expected = pd.DataFrame({
            'column_1' : [10, 20, 30]
        })
        
        # Act
        actual = multiply_by_10(df)
        
        # Assert
        assert_frame_equal(expected, actual)

