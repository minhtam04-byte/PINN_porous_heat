class DotDict(dict):
    """
    Giúp truy cập dictionary theo dạng object.attribute thay vì object['key'].
    Hỗ trợ đệ quy cho các dictionary lồng nhau.
    """
    def __init__(self, d=None):
        if d is None:
            d = {}
        super().__init__()
        for key, value in d.items():
            if isinstance(value, dict):
                value = DotDict(value)
            self[key] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __repr__(self):
        return f"DotDict({super().__repr__()})"