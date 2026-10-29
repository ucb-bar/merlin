"""Declared public names enrich source; inference and semantic edits refuse."""
import pytest
from merlin.targetgen.contract.linalg_iface import bind_logical_io_metadata

SOURCE='''module {func.func @forward(%x:tensor<2x3xi32>)->tensor<2x3xi32>{func.return %x:tensor<2x3xi32>}}'''
INPUT=[{'name':'source','shape':[2,3],'dtype':'i32','role':'input'}]
OUTPUT=[{'name':'result','shape':[2,3],'dtype':'i32'}]


def test_same_body_and_idempotent_metadata():
    text,receipt=bind_logical_io_metadata(SOURCE,INPUT,OUTPUT)
    again,_=bind_logical_io_metadata(text,INPUT,OUTPUT)
    assert again==text and receipt['body_transformations']==0
    assert 'merlin.logical_inputs = ["source"]'in text
    assert 'func.return %x' in text


@pytest.mark.parametrize('field,value',[('name','result'),('shape',[3,2]),('dtype','i8'),('role','output')])
def test_mismatched_declaration_refuses(field,value):
    with pytest.raises(ValueError):bind_logical_io_metadata(SOURCE,[{**INPUT[0],field:value}],OUTPUT)


def test_existing_inconsistent_names_refuse():
    text,_=bind_logical_io_metadata(SOURCE,INPUT,OUTPUT)
    with pytest.raises(ValueError):bind_logical_io_metadata(text,[{**INPUT[0],'name':'other'}],OUTPUT)


def test_wrong_arity_refuses():
    with pytest.raises(ValueError,match='arity'):bind_logical_io_metadata(SOURCE,INPUT+INPUT,OUTPUT)
