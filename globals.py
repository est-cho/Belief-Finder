from dataclasses import dataclass
import operator

TAG_BODY = 'body'
TAG_INPUT = 'input'
TAG_STATEMENT = 'statement'
TAG_PROP = 'prop'
TAG_VALUE = 'value'
TAG_TYPE = 'type'
TAG_INDEX = 'index'
TAG_TIME = 'time'
TAG_OP = 'op'

ATTRIB_NAME = 'name'
ATTRIB_LEFT = 'left'
ATTRIB_RIGHT = 'right'

VAL_TYPE_VAR = 'v'
VAL_TYPE_CONS = 'c'
VAL_FEILD_TYPE = ["color","speed","stop","step","deviation","integral","derivative","frontID","fronttime","frontspeed","frontdistance"]


OPERATORS = ["==", "<=", ">=", "<", ">", "!="]
OPERATOR_DICT = {"==": operator.eq, "<=": operator.le, ">=": operator.ge, 
                "<": operator.lt, ">": operator.gt, "!=": operator.ne}
SET_OPERATORS = [operator.__and__, operator.__or__]

@dataclass
class Value:
    type: str = ''
    index: int = 0
    time: int = None
    field: str = ''

    def copy(self):
        copy = Value()
        copy.type = self.type
        copy.index = self.index
        copy.time = self.time
        copy.field = self.field
        return copy

@dataclass
class Prop:
    def __init__(self, left = Value(), op:str = '==', right = Value(), is_unit:bool = True):
        self.left = left
        self.op = op
        self.right = right
        self.is_unit = is_unit
        if is_unit:
            if type(left) != Value or type(right) != Value or (not op in OPERATORS):
                raise ValueError('Unit Proposition Error')
        else:
            if type(left) != type(Prop) or type(right) != type(Prop) or (not op in SET_OPERATORS):
                raise ValueError('Composite Proposition Error')

    def copy(self):
        copy = Prop()
        copy.left = self.left.copy()
        copy.op = self.op
        copy.right = self.right.copy()
        copy.is_unit = self.is_unit
        return copy


@dataclass
class Statement:
    index: int = 0
    p_left: Prop = Prop()
    p_right: Prop = Prop()

    def copy(self):
        copy = Statement()
        copy.index = self.index
        copy.p_left = self.p_left.copy()
        copy.p_right = self.p_right.copy()
        return copy

@dataclass
class Parameter:
    time_range: int = 0
    variable_range: int = 0
    constant_range: int = 0

