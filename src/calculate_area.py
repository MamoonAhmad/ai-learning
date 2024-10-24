


def calculate_area(width, height):

    if(not isinstance(width, int) and not isinstance(width, float)):
        raise Exception("Invalid width")
    if(not isinstance(height, int) and not isinstance(height, float)):
        raise Exception("Invalid height")
    
    return width * height
