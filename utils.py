class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def convert_to_dotdict(d):
    if isinstance(d, dict):
        return DotDict({k: convert_to_dotdict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [convert_to_dotdict(i) for i in d]
    else:
        return d