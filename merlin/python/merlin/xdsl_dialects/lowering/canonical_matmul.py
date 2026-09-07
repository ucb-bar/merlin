"""Recognize the exact integer matrix multiply region, independently of tags."""
from xdsl.ir import Operation
from xdsl.ir.affine import AffineDimExpr


def is_integer_matmul(op: Operation) -> bool:
    """Admit canonical rank-two matmul only, including its generic spelling."""
    if op.name == "linalg.matmul":
        return True
    if op.name != "linalg.generic" or len(op.operands) != 3 or len(op.results) != 1:
        return False
    props = {**op.attributes, **op.properties}
    maps = props.get("indexing_maps")
    iters = props.get("iterator_types")
    if maps is None or len(maps) != 3 or iters is None:
        return False
    if [getattr(getattr(item, "data", None), "value", None) for item in iters] != [
            "parallel", "parallel", "reduction"]:
        return False
    for attr, expected in zip(maps, ((0, 2), (2, 1), (0, 1))):
        amap = attr.data
        if amap.num_dims != 3 or amap.num_symbols or len(amap.results) != 2:
            return False
        if not all(isinstance(expr, AffineDimExpr) and expr.position == position
                   for expr, position in zip(amap.results, expected)):
            return False
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    body = op.regions[0].blocks[0]
    if len(body.args) != 3:
        return False
    operations = list(body.ops)
    if not operations or operations[-1].name != "linalg.yield":
        return False
    if any(item.name not in {"arith.extsi", "arith.muli", "arith.addi", "linalg.yield"}
           for item in operations):
        return False
    value = operations[-1].operands[0]
    addition = value.owner
    if not isinstance(addition, Operation) or addition.name != "arith.addi":
        return False
    terms = list(addition.operands)
    if body.args[2] not in terms:
        return False
    product = terms[1 - terms.index(body.args[2])].owner
    if not isinstance(product, Operation) or product.name != "arith.muli":
        return False

    def base(value):
        producer = value.owner
        if isinstance(producer, Operation) and producer.name == "arith.extsi":
            return producer.operands[0]
        return value

    return {base(value) for value in product.operands} == set(body.args[:2])
