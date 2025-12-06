
from enum import Enum

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

def classify_option(ticker: str):
    call_maturities = ['A', 'B', 'C' ,'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    put_maturities = ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
    if ticker[5] in call_maturities:
        return OptionType.CALL
    elif ticker[5] in put_maturities:
        return OptionType.PUT
    else:
        raise ValueError("Ticker does not specify option type.")

colors = ['black', 'red', 'green', 'blue', 'olive', 'purple', 'orange', 'brown', 'pink', 'gray']