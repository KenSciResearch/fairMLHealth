'''
Back-end functions used throughout the library
'''


def is_dictlike(obj):
    dictlike = all([callable(getattr(obj, "keys", None)),
                    not hasattr(obj, "size")])
    return dictlike


class ValidationError(Exception):
    pass

