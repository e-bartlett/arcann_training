from typing import List


def replace_substring_in_list(
    input_list: List[str], substring_in: str, substring_out: str
) -> List[str]:
    """
    Replaces a specified substring with a new substring in each string in a given list.

    Args:
        input_list (List[str]): A list of input strings.
        substring_in (str): The substring to replace in the input strings.
        substring_out (str): The new substring to replace with.

    Returns:
        List[str]: A list of output strings with the specified substring replaced by the new substring.
    """

    # Use a list comprehension to create a new list with the updated strings
    output_list = [string.replace(substring_in, substring_out) for string in input_list]

    # Return the updated list
    return output_list


def delete_substring_from_list(input_list: List[str], substring: str) -> List[str]:
    """
    Delete all the strings in the input list that contain a given substring.

    Args:
        input_list (List[str]): The input list of strings.
        substring (str): The substring to look for in the input strings.

    Returns:
        List[str]: The output list of strings with all the strings containing the given substring removed.
    """

    # Use a list comprehension to create a new list with the updated strings
    output_list = [string for string in input_list if substring not in string]

    # Return the updated list
    return output_list
