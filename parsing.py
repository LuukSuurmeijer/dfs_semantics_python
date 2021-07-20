import nltk.sem.logic as fol
import string
import re
from typing import *


"""
This file contains functionality for parsing and dealing with first order logic expressions as provided by nltk.sem.logic.
"""

#Parsing first order logic.
PARSE = fol.Expression.fromstring
SYM = fol.Tokens()

#exisentially closes all free variales in an expression
def string_close_lambda(expression: fol.Expression) -> fol.Expression:
    return PARSE(str(expression).replace(f'{SYM.LAMBDA}', f'{SYM.EXISTS} '))

def string_close(expression: fol.Expression) -> fol.Expression:
    free_vars = expression.free()
    quantifiers =[f"{SYM.EXISTS} {str(var)}." for var in free_vars]
    new_string = ' '.join(quantifiers) + str(expression)
    return PARSE(new_string)

def get_term(expression: fol.Expression) -> fol.Expression:
    try:
        return get_term(expression.term)
    except AttributeError:
        return expression

def smart_replace(formula: str) -> str:
    alphabet_list = list(string.ascii_lowercase)
    reg = re.search(rf'((?:z(?:[0-9]*)(?: ))*(?:z(?:[0-9]*))\.)', formula)
    if reg:
        m = ' '.join(reg.groups(0)).strip('.').split(' ')
        newvars = [alphabet_list[23+i] for i in range(len(m))]
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
    match = re.search(rf'([a-z]+)(\((?:[a-z]+,)*{str(var)}(?:,[a-z]+)*\))', ex)
    return match[1], match[2]
