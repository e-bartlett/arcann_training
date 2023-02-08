def seconds_to_walltime(seconds: float) -> str:
    """Convert a time in seconds to a HH:MM:SS format

    Args:
        seconds (float): Float in seconds

    Returns:
        str: string in HH:MM:SS format
    """
    minutes, sec = divmod(seconds, 60)
    hour, minutes = divmod(minutes, 60)
    return "%d:%02d:%02d" % (hour, minutes, sec)
