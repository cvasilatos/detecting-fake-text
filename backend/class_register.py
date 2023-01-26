AVAILABLE_MODELS = {}


def register_api(name):
    def decorator(cls):
        print(f"Found API {cls} with name {name}")
        AVAILABLE_MODELS[name] = cls
        return cls
    return decorator
