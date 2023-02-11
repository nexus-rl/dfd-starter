DEFAULT_PORT = 50051

def resolve_port(address, port=None):
    has_scheme = '://' in address
    # I didn't check that these assumptions were true.
    if has_scheme and port is not None:
        raise ValueError('Cannot specify port with address that has scheme')

    if not has_scheme and port is None:
        print(f'No port specified, using default port: {DEFAULT_PORT}')
        port = DEFAULT_PORT

    if has_scheme:
        return address

    return f'{address}:{port}'
