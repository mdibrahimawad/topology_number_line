import numpy as np
import pytest
from numzig.ph_acceleration import lower_triangle,compare,sorted_diagram

def test_native_input_and_strict_equivalence():
    d=np.array([[0.,1.,2.],[1.,0.,3.],[2.,3.,0.]])
    assert np.array_equal(lower_triangle(d),np.array([1,2,3],dtype=np.float32))
    assert compare([[2,3],[0,1]],[[0,1],[2,3]],3)['passed']
    for bad in [np.array([[0.,2.],[1.,0.]]),np.array([[1.,2.],[2.,0.]])]:
        with pytest.raises(ValueError):lower_triangle(bad)
    with pytest.raises(ValueError):compare([[0,1]],[],1)
    with pytest.raises(ValueError):compare([[0,1]],[[0,1.1]],1)
    with pytest.raises(ValueError):sorted_diagram([[0,np.inf]])
