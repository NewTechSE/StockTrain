
def get_period_from_interval(interval: str) -> str:
    if interval == '1m':
        return '1d'
    elif interval == '60m':
        return '1mo'
    else:
        return '6mo'


