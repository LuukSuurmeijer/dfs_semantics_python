import nltk.sem.logic as fol
import string
import re
from typing import *


"""
This file contains functionality for parsing and dealing with first order logic expressions as provided by nltk.sem.logic.
"""

#Parsing first order logic.
PARSE = fol.Expression.fromstring
T_PARSE = fol.Type.fromstring
SYM = fol.Tokens()

def flip(expression: fol.Expression):
    flipped_args = [str(arg) for arg in list(reversed(expression.uncurry()[1]))]
    func = expression.uncurry()[0]
    newstring = f"{str(func)}({','.join(flipped_args)})"
    return newstring

#exisentially closes all free variales in an expression
def string_close_lambda(expression: fol.Expression) -> fol.Expression:
    return PARSE(str(expression).replace(f'{SYM.LAMBDA}', f'{SYM.EXISTS} '))

def string_close(expression: fol.Expression) -> fol.Expression:
    free_vars = sorted(list(expression.free())) #predicates need to always come first, hence sort the variables alphabetically
    quantifiers =[f"{SYM.EXISTS} {str(var)}." for var in free_vars]
    new_string = ' '.join(quantifiers) + str(expression)
    return PARSE(new_string)

def get_term(expression: fol.Expression) -> fol.Expression:
    try:
        return get_term(expression.term)
    except AttributeError:
        return expression

def get_property_type(t : fol.ComplexType, typelist: list) -> list:
    while not t.matches(T_PARSE('<e, t>')):
        try:
            typelist.append(t.first)
            t = t.second
        except AttributeError:
            return False

    return typelist


def get_arguments(expression: fol.Expression):
    if isinstance(expression, fol.ApplicationExpression):
        return expression.uncurry()

def smart_replace(formula: str) -> str:
    alphabet_list = list(string.ascii_lowercase)
    #first order variables
    reg = re.findall(rf'((?:z(?:[0-9]+)(?: ))*(?:z(?:[0-9]+))(?:\.*))', formula)
    if reg:
        m = [item.strip('.') for item in list(set(reg))]
        newvars = [alphabet_list[((23+i) % (len(alphabet_list)))] for i in range(len(m))]
        for i in range(len(m)):
            formula = formula.replace(m[i], newvars[i])

    #second order variables
    reg = re.findall(rf'((?:F(?:[0-9]+)(?: ))*(?:F(?:[0-9]+))(?:\.*))', formula)
    alphabet_list = list(string.ascii_uppercase)
    if reg:
        m = [item.strip('.') for item in list(set(reg))]
        newvars = [alphabet_list[((15+i) % (len(alphabet_list)))] for i in range(len(m))]
        for i in range(len(m)):
            formula = formula.replace(m[i], newvars[i])
    return formula

def deconstruct_expression(expression: str) -> str:
    """
    Function to remove the outermost quantifier in a string formatted like an nltk.sem.logic.Expression
    """
    match = re.match(rf'({SYM.EXISTS}|{SYM.ALL}) (([a-z][0-9]*)(?: ))*(([a-z][0-9]*)(?:\.))', expression)
    variablebinder = match[0].strip('.').split(' ')
    quantifier = variablebinder[0]
    variables = variablebinder[1:]

    if len(variables) > 1: #if there are more than 1 variable, we need to only delete the first one
        new_variables = variables[1:]
        new_quantifier = quantifier + ' ' + ' '.join(new_variables) #this is bad, but what do?
        expression = expression.strip(f'{variablebinder}')
        return new_quantifier + expression

    else: #if theres only one variable, we remove the whole quantifier
        remove = ' '.join(variablebinder) + '.'
        return expression.split(f'{remove}')[-1]

def find_predicate(ex: str, var: fol.Variable) -> str:
    parsed_var = PARSE(str(var))
    if isinstance(parsed_var, fol.FunctionVariableExpression):
        match = re.search(rf'({str(var)})(\((?:[a-z]+[0-9]*,*)+)\)', ex)
    else:
        match = re.search(rf'([A-z]+)(\((?:[a-z]+[0-9]*,)*{str(var)}(?:,[a-z]+[0-9]*)*\))', ex)
    return match[1], match[2]

def unscramble_bullshit(ex: fol.Expression):
    pass
