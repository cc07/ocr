import os
import cv2
import random
import json
from PIL import Image, ImageFilter

from computer_text_generator import ComputerTextGenerator
try:
    from handwritten_text_generator import HandwrittenTextGenerator
except ImportError as e:
    print('Missing modules for handwritten text generation.')
from background_generator import BackgroundGenerator
from distorsion_generator import DistorsionGenerator

dictionary = {}

class FakeTextDataGenerator(object):
    @classmethod
    def generate(cls, index, text, font, out_dir, height, extension, skewing_angle, random_skew, blur, random_blur, background_type, distorsion_type, distorsion_orientation, is_handwritten, name_format, text_color=-1):


        decode = {"a":"1","b":"2","c":"3","d":"4","e":"5","f":"6","g":"7",
          "h":"8","i":"9","j":"0","k":"0","l":"1","m":"1","n":"0",
          "o":"0","p":"1","q":"2","r":"3","s":"4","t":"5","u":"6",
          "v":"7","w":"8","x":"9","y":"8","z":"7"}
        output=""
        for char in text:
            if (char.islower()):
                output += decode[char]
            else:
                output += char


        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            image = HandwrittenTextGenerator.generate(text)
        else:
            image = ComputerTextGenerator.generate(text, font, text_color)

        random_angle = random.randint(0-skewing_angle, skewing_angle)

        rotated_img = image.rotate(skewing_angle if not random_skew else random_angle, expand=1)

        #############################
        # Apply distorsion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img # Mind = blown
        elif distorsion_type == 1:
            distorted_img = DistorsionGenerator.sin(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )
        elif distorsion_type == 2:
            distorted_img = DistorsionGenerator.cos(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )
        else:
            distorted_img = DistorsionGenerator.random(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )

        new_text_width, new_text_height = distorted_img.size

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background = BackgroundGenerator.gaussian_noise(new_text_height + 10, new_text_width + 10)
        elif background_type == 1:
            background = BackgroundGenerator.plain_white(new_text_height + 10, new_text_width + 10)
        elif background_type == 2:
            background = BackgroundGenerator.quasicrystal(new_text_height + 10, new_text_width + 10)
        else:
            background = BackgroundGenerator.picture(new_text_height + 10, new_text_width + 10)

        mask = distorted_img.point(lambda x: 0 if x == 255 or x == 0 else 255, '1')

        background.paste(distorted_img, (5, 5), mask=mask)

        ##################################
        # Resize image to desired format #
        ##################################
        new_width = float(new_text_width + 10) * (float(height) / float(new_text_height + 10))
        image_on_background = background.resize((int(new_text_width), height), Image.ANTIALIAS)

        final_image = image_on_background.filter(
            ImageFilter.GaussianBlur(
                radius=(blur if not random_blur else random.randint(0, blur))
            )
        )

        #####################################
        # Generate name for resulting image #
        #####################################

        dictionary[index] = output
        if (index==99):
            with open('label.json','w') as outfile:
                json.dump(dictionary, outfile)

        if name_format == 0:
            image_name = '{}_{}.{}'.format(output, str(index), extension)
        elif name_format == 1:
            image_name = '{}_{}.{}'.format(str(index), output, extension)
        else:
            print('{} is not a valid name format. Using default.'.format(name_format))
            image_name = '{}_{}.{}'.format(output, str(index), extension)

        # Save the image
        final_image.convert('RGB').save(os.path.join(out_dir, image_name))
