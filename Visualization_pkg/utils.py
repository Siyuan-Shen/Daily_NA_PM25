import toml
import datetime

def return_sign(number):
    if number < 0.0:
        return '-'
    elif number == 0.0:
        return ''
    else:
        return '+'  