import scipy
import numpy 
import cv2
from cv2.typing import MatLike
import colour
import time
import os

# testing generative functions in python

# importing dataset
emoji_path = r'img\emojis\image\Apple'
all_files = os.listdir(emoji_path)

# filter image files by type
image_files_names = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

# init_seed = numpy.random.randint(1,1816, size=(1,100))

t_size = tuple[int, int]
t_pos = tuple[int,int]

# Alias   emoji,   size,       pos,     score
Species = tuple[int, t_size, t_pos, float]


execution_times = {}

def calculate_execution_time(identifier):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            if identifier in execution_times:
                execution_times[identifier].append(execution_time)
            else:
                execution_times[identifier] = [execution_time]
            print(f"Execution time for {identifier}: {execution_time} seconds")
            return result
        return wrapper
    return decorator


def fmt_img(rgb_image:MatLike)->MatLike:
    '''
    formats image into a matrix of floats to be used by a delta e function
    '''
    return cv2.cvtColor(rgb_image.astype(numpy.float32) / 255, cv2.COLOR_RGB2Lab)


@calculate_execution_time("score_img")
def scoring_image(base_image:MatLike, new_image:MatLike)->float:
    """ 
     this method takes in a base image and a new image, 
     compares the two, then outputs a score based on the 
     difference between the two images, a higher score 
     should correspond in a two images that have similar 
     color values at their positions

    @param base_image
    @param new_image

    @return score
    """
    dE = colour.delta_E(fmt_img(base_image), fmt_img( new_image))

    avg = float(numpy.mean(dE))

    return avg

def overlay_transparent(background_img, img_to_overlay_t, pos:t_pos, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2
    
    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None
    
    @return     Background image with overlay on top
    """
    
    bg_img = background_img.copy()
    x = pos[0]
    y = pos[1]
    
    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB 
    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))
    
    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a,5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
    
    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

    # Update the original image with our new ROI
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

    return bg_img

em_path = 'img/emojis/image/Apple/'

image1_rgb = cv2.imread('img/target.png',-1)
image2_rgb = cv2.imread('img/start.png',-1)
image3_rgb = (cv2.imread(f'{em_path}1.png',-1))

added_image = overlay_transparent(image2_rgb, image3_rgb, (100, 100), (500,500))

print(scoring_image(image1_rgb, added_image))
print(scoring_image(image1_rgb, image2_rgb))
 
# cv2.imwrite('img/result_lab.png', added_image)

def new_species(pos:t_pos, size:t_size, emoji:int, score:float)->Species:
    spec:Species = (emoji, size, pos, score )
    return spec 

def load_emoji(id:int|None = None):
    if id ==None:
        r = numpy.random.randint(1,1816)
    else:
        r = id
    emoji = (cv2.imread(f'{em_path}{r}.png',-1))
    return emoji

def get_size(seed:t_size|None = None)->t_size:
    max_height:int = 500
    max_width:int  = 500
    xbound:int = max_width//10
    ybound:int = max_height//10
    if seed == None:
        return (numpy.random.randint(max_width),numpy.random.randint(max_height))
    else:
        #Allow for +-10% difference from seed 
        temp:t_size = (numpy.random.randint(-xbound, xbound)+seed[0],numpy.random.randint(-ybound,ybound)+seed[1])
        if temp[0] < 0:
            temp:t_size = (0,temp[1])
        elif temp[0] > max_width:
            temp:t_size = (max_width, temp[1])
        if temp[1] < 0:
            temp:t_size = (temp[0], 0)
        elif temp[1] > max_height:
            temp:t_size = (temp[0], max_height)
    
        return temp

def get_pos(seed:t_pos|None = None)->t_pos:
    max_height:int = 1080
    max_width:int  = 1728
    xbound:int = max_width//10
    ybound:int = max_height//10
    if seed == None:
        return (numpy.random.randint(max_width),numpy.random.randint(max_height))
    else:
        #Allow for +-10% difference from seed 
        temp:t_size = (numpy.random.randint(-xbound, xbound)+seed[0],numpy.random.randint(-ybound,ybound)+seed[1])
        if temp[0] < 0:
            temp:t_size = (0,temp[1])
        elif temp[0] > max_width:
            temp:t_size = (max_width, temp[1])
        if temp[1] < 0:
            temp:t_size = (temp[0], 0)
        elif temp[1] > max_height:
            temp:t_size = (temp[0], max_height)
    
        return temp    

def process(base:MatLike, goal:MatLike)->Species:
    '''
    input: starting image, target image
    output: species

    this method should randomly place a random emoji on the screen, 
    determine the difference between the new image and the target image.
    '''
    
    #TODO: Get random position
    
    #// TODO: Get random size
    size  = get_size()
    
    
    #// TODO: Get random emoji 
    
    emoji = load_emoji()
    
    #TODO: add emoji
    overlay_transparent(base, emoji, pos, size)
    
    #TODO: score
    
    
    return

def epoch():
    # select the top scoring species
    pass

def advance():
    # cull the bottom 50%, diversify the top 50%
    pass

## Multiprocessing

num_processes = 5
process_results = []

# Create a list to store process instances
processes = []

# Create and start processes
for i in range(num_processes):
    process = multiprocessing.Process(target=lambda i=i: process_results.append(process_function(i)))
    processes.append(process)
    process.start()

# Wait for all processes to finish
for process in processes:
    process.join()

# Print results from all processes
print("All processes have finished. Results:")
for i, result in enumerate(process_results):
    print(f"Process {i}: {result}")

