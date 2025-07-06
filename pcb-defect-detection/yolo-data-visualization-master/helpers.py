import random

# Function to generate a random color
def generate_color():
    # Generate random values for red, green, and blue channels
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    
    # Ensure sufficient contrast for visualization
    if (r + g + b) / 3 < 128:
        return generate_color()  # If color is too dark, try again
    else:
        return (r, g, b)


