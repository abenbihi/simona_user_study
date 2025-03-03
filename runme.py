import argparse
import json
import os
import string

import cv2
import numpy as np


QUESTIONS = {
    "Q1": "Was an object removed from this image? (y/n)",
    "Q2": "Was an object removed from this area? (y/n)",
    "Q3": "In which image is the object best removed?", # 4 methods,
    "Q4": "What object was there?",
    "Q5": "In which image is the object best removed?", # 2 methods,
}

#OBJECTS = [l.strip() for l in open("data/labels.txt", "r").readlines()]
OBJECTS = {}
OBJECTS["garden"] = ["table", "chair", "car", "plant", "barbecue", "dog"]
OBJECTS["kitchen"] = ["eggbox" "tray", "plant", "table", "pasta", "gloves",
        "ball", "roll", "vase", "salt", "glass", "plate", "mixer", "candle"
        ]

def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def resize_horizontal(h1, w1, h2, w2, target_height):
    scale_to_align = float(h1) / h2
    current_width = w1 + w2 * scale_to_align
    scale_to_fit = target_height / h1
    target_w1 = int(w1 * scale_to_fit)
    target_w2 = int(w2 * scale_to_align * scale_to_fit)
    target_h = int(target_height)
    return (target_w1, target_h), (target_w2, target_h), scale_to_fit, scale_to_fit * scale_to_align, [target_w1, 0]


def draw_box(box, img, color=(0,255,0)):
    xmin, ymin, xmax, ymax = box
    p0 = (int(xmin), int(ymin)) # top left corner
    p2 = (int(xmax), int(ymax)) # bottom right corner

    p1 = [p0[0], p2[1]]
    xmin = p0[0]
    ymin = p0[1]

    cv2.rectangle(img, p0, p2, color, 2, 8)


def get_box_from_binary_mask(mask):
    """
    Returns:
        box: [xmin, ymin, xmax, ymax] that defines the range of the box
    """
    # TODO: assuming the mask is 255 on the object and 0 elsewhere
    assert(mask.ndim == 2) # read the mask in black and white
    l, c = np.where(mask==255)
    xmin = np.min(c)
    xmax = np.max(c)
    ymin = np.min(l)
    ymax = np.max(l)

    return [xmin, ymin, xmax, ymax]


def save_results(output_path, output):
    with open(output_path, "w") as f:
        json.dump(output, f)
    

def main_binary(args, question_tag):
    """Ask a binary question to the user that answers yes / no."""
    # Read image to list to display
    image_list_path = "data/image_list.txt"
    image_names = [l.strip().split(" ") for l in open(image_list_path,"r").readlines() if l[0] != "#"]

    # Read previous user's response if they are resuming the experiment
    output_path = "answers_%s.json"%args.username
    output = None
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            output = json.load(f)
    else:
        # initialize the answers' entries
        output = {}
        #for q_tag, _ in QUESTIONS.items():
        q_tag = question_tag
        output[q_tag] = {}
        for image_name in image_names:
            image_key = image_name[0]
            output[q_tag][image_key] = -1

    if question_tag not in output:
        output[question_tag] = {}

    # Display the question
    print(QUESTIONS[question_tag])
    print("INSTRUCTIONS")
    print("\nPress 'y' for yes / 'n' for no")
    print("To come back to the previous image, press 'p'")
    print("To skip an image, press 'x'")
    print("To quit, press 'q'")

    idx = 0
    while True:
        if idx == len(image_names):
            print("You have reached the end of the images")
            break

        print(idx)
        # Read next image
        rand_gt = np.random.rand()
        is_object_removed = None
        if rand_gt < 0.5:
            # show render before removal
            is_object_removed = 0
            image_name = image_names[idx][1]
        else:
            # show_render after removal 
            is_object_removed = 1
            image_name = image_names[idx][0]
        check_path(image_name)
        img = cv2.imread(image_name)

        if rand_gt < 0.5:
            img[0:100,0:100] = 0

        # Read mask associated to it to derive a bounding box from it to guide
        # the human's eye
        if question_tag == "Q2": # draw a box
            # TODO: set the mask path (uncomment this)
            #mask_path = None
            #mask = cv2.imread(mask_path, 0)

            # TODO: mock mask in the mean time (comment this)
            h, w = img.shape[:2]
            mask = np.zeros((h,w), dtype=np.uint8)
            cv2.circle(mask, (w//2,h//2), min(h,w)//2, 255, -1)

            box = get_box_from_binary_mask(mask)

            # Draw the box on the image
            draw_box(box, img)

        #if args.target_height > 0:
        #    h, w = img.shape[:2]
        #    target, target, scale, scale, offset = resize_horizontal(h, w, h, w, args.target_height)
        #    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) 

        cv2.imshow("img", img)
        #cv2.imshow("mask", mask)
        key = cv2.waitKey(0) & 0xFF

        if key not in [ord("y"), ord("n"), ord("p"), ord("l"), ord("q")]:
            print("Wrong key, please press one of the following keys y / n / p / l / q")
            print("See instructions above")
            continue

        # For binary questions

        image_key = image_names[idx][0]
        if key == ord("y"):
            user_answer = 1
            output[question_tag][image_key] = (user_answer, is_object_removed) 
            idx += 1
            save_results(output_path, output)
            continue

        if key == ord("n"):
            user_answer = 0
            output[question_tag][image_key] = (user_answer, is_object_removed) 
            idx += 1
            save_results(output_path, output)
            continue

        if key == ord("p"):
            if idx == 0:
                print("This is already the first image")
                continue
            idx -= 1
            continue

        if key == ord("x"):
            if idx == len(image_names)-1:
                print("This is already the last image")
                continue
            idx += 1
            continue

        if key == ord("q"):
            # Save results
            break


def main_multi_method(args, question_tag, num_methods=4):
    """Show results from 4 methods and ask user to pick one (a,b,c,d)."""
    alphabet_list = list(string.ascii_lowercase)
    print(alphabet_list)

    # Set of keys the user can pick
    authorized_keys = [ord("p"), ord("x"), ord("q")]
    alphabet_choice = alphabet_list[:num_methods] # string format
    alphabet_keys = [ord(l) for l in alphabet_choice] # ascii encoding
    authorized_keys += alphabet_keys
    print(authorized_keys)

    # Read image to list to display
    image_list_path = "data/multi_method_list.txt"
    #image_list_path = "data/multi_method_list3.txt" # for 3 methods
    image_tuples = [l.strip().split(" ") for l in open(image_list_path,"r").readlines()]
    if len(image_tuples[0]) != num_methods:
        raise ValueError("@Simona, change the num_methods param")

    # Read previous user's response if they are resuming the experiment
    output_path = "answers_%s.json"%args.username
    output = None
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            output = json.load(f)
    else:
        # initialize the answers' entries
        output = {}
        q_tag = question_tag
        #for q_tag, _ in QUESTIONS.items():
        output[q_tag] = {}
        for image_tuple in image_tuples:
            #image_key = image_tuple[0]
            image_key = " ".joint(image_tuple)
            output[q_tag][image_key] = -1

    if question_tag not in output:
        output[question_tag] = {}

    # Format the question
    question_choices = {}
    question_str = ""
    for idx, letter in enumerate(alphabet_choice):
        question_str += "%s.\n"%(letter)
        question_choices[letter] = idx
    #print("gt object: ", gt_object_name)
    #print(QUESTIONS[question_tag])
    #print(question_str)
    #exit(1)

    # Display the question
    print(QUESTIONS[question_tag])
    print("\nPress the letter that appears at the top left corner of the image: ")
    #%s."%(
    #    " / ".join(alphabet_choice)))
    print(question_str)
    print("To come back to the previous image, press 'p'")
    print("To skip an image, press 'x'")
    print("To quit, press 'q'")

    image_idx = 0
    while True:
        if image_idx == len(image_tuples):
            print("You have reached the end of the images")
            break

        # Read next image
        image_tuple = image_tuples[image_idx]
        images = []
        for tuple_idx, image_name in enumerate(image_tuple):
            #print(image_name)
            check_path(image_name)
            img = cv2.imread(image_name)

            # Read mask associated to it to derive a bounding box from it to guide
            # the human's eye
            # TODO: set the mask path (uncomment me)
            #mask_path1 = None
            #mask1 = cv2.imread(mask_path1, 0)

            # TODO: mock mask in the mean time (comment me)
            h, w = img.shape[:2]
            mask = np.zeros((h,w), dtype=np.uint8)
            cv2.circle(mask, (w//2,h//2), min(h,w)//2, 255, -1)

            box = get_box_from_binary_mask(mask)

            # Draw the box on the image
            draw_box(box, img)

            # Draw letter
            textSize = 2.0
            pos = (0, 50)
            text = alphabet_choice[tuple_idx]
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        textSize,(0,200,0),2, cv2.LINE_AA)

            images.append(img)

        #if args.target_height > 0:
        #    h, w = img.shape[:2]
        #    target, target, scale, scale, offset = resize_horizontal(h, w, h, w, args.target_height)
        #    img1 = cv2.resize(img1, (w,h), interpolation=cv2.INTER_AREA) 
        #    img2 = cv2.resize(img2, (w,h), interpolation=cv2.INTER_AREA) 

        if num_methods == 3:
            h,w = images[0].shape[:2]
            tmp = np.zeros((h,w,3), dtype=np.uint8)
            images.append(tmp)

        if num_methods == 4 or num_methods == 3:
            line1 = np.hstack(images[:2])
            line2 = np.hstack(images[2:])
            out = np.vstack((line1, line2))
            h,w = images[0].shape[:2]
            out[:,w-2:w+2] = 0
            out[h-2:h+2,:] = 0
        elif num_methods == 2:
            out = np.hstack(images[:2])
            h,w = images[0].shape[:2]
            out[:,w-2:w+2] = 0
        else:
            raise ValueError(
                    "Handling %d methods is not implemented, num_methods must be in {2,3,4}")
    
        cv2.imshow("img", out)
        #cv2.imshow("mask", mask)
        key = cv2.waitKey(0) & 0xFF

        if key not in authorized_keys:
            #tmp = "y / n / p / l / q / " + " / ".join(alphabet_choice) 
            tmp = " / ".join(alphabet_choice) 
            print("Wrong key, please press one of the following keys: %s"%tmp)
            print("See instructions above")
            continue

        image_key = " ".join(image_tuple)
        if key in alphabet_keys:
            print(chr(key))
            output[question_tag][image_key] = question_choices[chr(key)]
            image_idx += 1
            save_results(output_path, output)
            continue

        if key == ord("p"):
            if image_idx == 0:
                print("This is already the first image")
                continue
            image_idx -= 1
            continue

        if key == ord("x"):
            if image_idx == len(image_names)-1:
                print("This is already the last image")
                continue
            image_idx += 1
            continue

        if key == ord("q"):
            # Save results
            break


def main_binary_method(args, question_tag):
    """Show results from 2 methods and ask user to pick one (left / right)."""
    # Read image to list to display
    image_list_path = "data/pair_list.txt"
    image_pairs = [l.strip() for l in open(image_list_path,"r").readlines()]

    # Read previous user's response if they are resuming the experiment
    output_path = "answers_%s.json"%args.username
    output = None
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            output = json.load(f)
    else:
        # initialize the answers' entries
        output = {}
        q_tag = question_tag
        #for q_tag, _ in QUESTIONS.items():
        output[q_tag] = {}
        for image_pair in image_pairs:
            output[q_tag][image_pair] = -1

    if question_tag not in output:
        output[question_tag] = {}

    # Display the question
    print(QUESTIONS[question_tag])
    print("\nPress 'l' / 'r' for the left / right image")
    print("To come back to the previous image, press 'p'")
    print("To skip an image, press 'x'")
    print("To quit, press 'q'")

    idx = 0
    while True:
        if idx == len(image_pairs):
            print("You have reached the end of the images")
            break

        print(idx)
        # Read next image
        image_pair = image_pairs[idx]
        image_name1, image_name2 = image_pair.split(" ")
        #print(image_name1)
        #print(image_name2)
        check_path(image_name1)
        check_path(image_name2)
        img1 = cv2.imread(image_name1)
        img2 = cv2.imread(image_name2)

        # Read mask associated to it to derive a bounding box from it to guide
        # the human's eye
        # TODO: set the mask path (uncomment me)
        #mask_path1 = None
        #mask_path2 = None
        #mask1 = cv2.imread(mask_path1, 0)
        #mask2 = cv2.imread(mask_path2, 0)

        # TODO: mock mask in the mean time (comment me)
        h, w = img1.shape[:2]
        mask1 = np.zeros((h,w), dtype=np.uint8)
        cv2.circle(mask1, (w//2,h//2), min(h,w)//2, 255, -1)
        mask2 = mask1


        box1 = get_box_from_binary_mask(mask1)
        box2 = get_box_from_binary_mask(mask2)

        # Draw the box on the image
        draw_box(box1, img1)
        draw_box(box2, img2)

        #if args.target_height > 0:
        #    h, w = img.shape[:2]
        #    target, target, scale, scale, offset = resize_horizontal(h, w, h, w, args.target_height)
        #    img1 = cv2.resize(img1, (w,h), interpolation=cv2.INTER_AREA) 
        #    img2 = cv2.resize(img2, (w,h), interpolation=cv2.INTER_AREA) 
    
        print(img1.shape)
        print(img2.shape)
        out = np.hstack((img1, img2))
        h,w = img1.shape[:2]
        out[:,w-2:w+2] = 0

        cv2.imshow("img", out)
        #cv2.imshow("mask", mask)
        key = cv2.waitKey(0) & 0xFF

        if key not in [ord("y"), ord("n"), ord("p"), ord("x"), ord("q"), ord("l"), ord("r")]:
            print("Wrong key, please press one of the following keys y / n / p / x / l / r")
            print("See instructions above")
            continue

        # For binary questions
        if key == ord("l"):
            output[question_tag][image_pair] = "left"
            idx += 1
            save_results(output_path, output)
            continue

        if key == ord("r"):
            output[question_tag][image_pair] = "right"
            idx += 1
            save_results(output_path, output)
            continue

        if key == ord("p"):
            if idx == 0:
                print("This is already the first image")
                continue
            idx -= 1
            continue

        if key == ord("x"):
            if idx == len(image_names)-1:
                print("This is already the last image")
                continue
            idx += 1
            continue

        if key == ord("q"):
            # Save results
            break


## TODO: still not smooth
#def main_prompt(args, question_tag):
#    """ """
#    # Read image to list to display
#    image_list_path = "data/image_list.txt"
#    image_names = [l.strip() for l in open(image_list_path,"r").readlines()]
#
#    # Read previous user's response if they are resuming the experiment
#    output_path = "answers_%s.json"%args.username
#    output = None
#    if os.path.exists(output_path):
#        with open(output_path, "r") as f:
#            output = json.load(f)
#    else:
#        # initialize the answers' entries
#        output = {}
#        for q_tag, _ in QUESTIONS.items():
#            output[q_tag] = {}
#            for image_name in image_names:
#                output[q_tag][image_name] = -1
#
#    # Display the question
#    print(QUESTIONS[question_tag])
#    print("\nEnter your answer then press 'k'")
#    #Press y for yes / n for no")
#    #print("To come back to the previous image, press 'p'")
#    #print("To skip an image, press 'l'")
#    print("To quit, press 'q'")
#
#    idx = 0
#    while True:
#        if idx == len(image_names):
#            print("You have reached the end of the images")
#            break
#
#        print(idx)
#        # Read next image
#        image_name = image_names[idx]
#        img = cv2.imread(image_name)
#        print(img.shape)
#
#        # Read mask associated to it to derive a bounding box from it to guide
#        # the human's eye
#        # TODO: set the mask path
#        #mask_path = None
#        #mask = cv2.imread(mask_path, 0)
#
#        # TODO: mock mask in the mean time
#        h, w = img.shape[:2]
#        mask = np.zeros((h,w), dtype=np.uint8)
#        cv2.circle(mask, (w//2,h//2), min(h,w)//2, 255, -1)
#        box = get_box_from_binary_mask(mask)
#
#        # Draw the box on the image
#        draw_box(box, img)
#
#        #if args.target_height > 0:
#        #    h, w = img.shape[:2]
#        #    target, target, scale, scale, offset = resize_horizontal(h, w, h, w, args.target_height)
#        #    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) 
#
#        cv2.imshow("img", img)
#        cv2.waitKey(10)
#
#        # TODO: For text prompts
#        input_str = "%s\nEnter the object name and press 'k'\n"%QUESTIONS[question_tag]
#        object_name = input(input_str)
#        output[question_tag][image_name] = object_name
#
#        cv2.imshow("img", img)
#        key = cv2.waitKey(0) & 0xFF
#
#        if key == ord("k"):
#            idx += 1
#            save_results(output_path, output)
#            continue
#
#        if key == ord("p"):
#            if idx == 0:
#                print("This is already the first image")
#                continue
#            idx -= 1
#            continue
#
#        if key == ord("l"):
#            if idx == len(image_names)-1:
#                print("This is already the last image")
#                continue
#            idx += 1
#            continue
#
#        if key == ord("q"):
#            # Save results
#            break


def main_choice(args, question_tag, num_choices=4):
    """Ask user if they can see one of the objects in the specified list (a,b,c,d, ...)."""
    alphabet_list = list(string.ascii_lowercase)
    print(alphabet_list)

    # Set of keys the user can pick
    authorized_keys = [ord("p"), ord("x"), ord("q")]
    alphabet_choice = alphabet_list[:num_choices] # string format
    alphabet_keys = [ord(l) for l in alphabet_choice] # ascii encoding
    authorized_keys += alphabet_keys
    print(authorized_keys)

    # Read image to list to display
    image_list_path = "data/image_list.txt"
    image_names = [l.strip().split(" ")[0] for l in open(image_list_path,"r").readlines() if l[0] != "#"]
    
    # TODO: specify the scene
    scene = image_names[0].split("/")[2]
    assert(scene in OBJECTS)

    # Read previous user's response if they are resuming the experiment
    output_path = "answers_%s.json"%args.username
    output = None
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            output = json.load(f)
    else:
        # initialize the answers' entries
        output = {}
        #for q_tag, _ in QUESTIONS.items():
        q_tag = question_tag
        output[q_tag] = {}
        for image_name in image_names:
            output[q_tag][image_name] = -1

    if question_tag not in output:
        output[question_tag] = {}

    # Display the question
    print(QUESTIONS[question_tag])
    print("\nPress the letter associated with the object.")
    print("a. cat")
    print("b. turtle")
    print("c. dog")
    print("For example, to pick the 'cat', press 'a', to pick the dog press 'c' ...")
    print("To come back to the previous image, press 'p'")
    print("To skip an image, press 'x'")
    print("To quit, press 'q'\n")

    image_idx = 0
    while True:
        if image_idx == len(image_names):
            print("You have reached the end of the images")
            break

        # Read next image
        image_name = image_names[image_idx]
        img = cv2.imread(image_name)

        # Read mask associated to it to derive a bounding box from it to guide
        # the human's eye
        # TODO: set the mask path (uncomment me)
        #mask_path = None
        #mask = cv2.imread(mask_path, 0)

        # TODO: mock mask in the mean time (comment me)
        h, w = img.shape[:2]
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.circle(mask, (w//2,h//2), min(h,w)//2, 255, -1)
        box = get_box_from_binary_mask(mask)

        # Draw the box on the image
        draw_box(box, img)

        #if args.target_height > 0:
        #    h, w = img.shape[:2]
        #    target, target, scale, scale, offset = resize_horizontal(h, w, h, w, args.target_height)
        #    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) 

        #exit(1)

        # Generate set of choices
        gt_object_name = image_name.split("/")[3] # The actual object
        # A set of other objects (DO NOT EDIT unless you are sure of what you
        # are doing)
        # Sampling elements without replacement
        indices = np.arange(len(OBJECTS[scene]))
        np.random.shuffle(indices)
        indices = indices[:num_choices+1] # keep the +1
        print(indices)
        print(OBJECTS[scene])
        other_objects = [OBJECTS[scene][idx] for idx in indices ] 
        #if OBJECTS[scene][idx] !=
        #        gt_object_name]
        other_objects = other_objects[:num_choices]

        # Pick the rank of the gt object
        gt_idx = np.random.randint(num_choices)
        
        # Format the question
        question_choices = {}
        question_str = ""
        for idx, other_object in enumerate(other_objects):
            if idx == gt_idx:
                question_str += "%s. %s\n"%(alphabet_list[idx], gt_object_name)
                question_choices[alphabet_list[idx]] = gt_object_name
            else:
                question_str += "%s. %s\n"%(alphabet_list[idx], other_object)
                question_choices[alphabet_list[idx]] = other_object
        #print("gt object: ", gt_object_name)
        print(QUESTIONS[question_tag])
        print(question_str)
        #exit(1)

        # Prepare the log
        output[question_tag][image_name] = {}
        output[question_tag][image_name]["gt"] = gt_object_name
        output[question_tag][image_name]["choices"] = []
        for idx, other_object in enumerate(other_objects):
            if idx == gt_idx:
                output[question_tag][image_name]["choices"].append(gt_object_name)
                #output[question_tag][image_name][alphabet_list[idx]] = gt_object_name
            else:
                #output[question_tag][image_name][alphabet_list[idx]] = other_object
                output[question_tag][image_name]["choices"].append(other_object)
        #print(output[question_tag][image_name]) # print the choices + gt

        cv2.imshow("img", img)
        cv2.imshow("mask", mask)
        key = cv2.waitKey(0) & 0xFF

        if key not in authorized_keys:
            #tmp = "y / n / p / l / q / " + " / ".join(alphabet_choice) 
            tmp = " / ".join(authorized_keys) 
            print("Wrong key, please press one of the following keys: %s"%tmp)
            print("See instructions above")
            continue

        if key in alphabet_keys:
            print(chr(key))
            output[question_tag][image_name]["user"] = question_choices[chr(key)]
            image_idx += 1
            save_results(output_path, output)
            continue

        if key == ord("y"):
            image_idx += 1
            save_results(output_path, output)
            continue

        if key == ord("n"):
            output[question_tag][image_name] = 0
            image_idx += 1
            save_results(output_path, output)
            continue

        if key == ord("p"):
            if image_idx == 0:
                print("This is already the first image")
                continue
            image_idx -= 1
            continue

        if key == ord("x"):
            if image_idx == len(image_names)-1:
                print("This is already the last image")
                continue
            image_idx += 1
            continue

        if key == ord("q"):
            # Save results
            break


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, 
            help="Your first name in lower case and without space",
            required=True)
    parser.add_argument("--target_height", type=int, default=600,
            help="To control the image size. Decrease to get smaller images.")
    parser.add_argument("--question", type=str, 
            help="Question to answer", required=True)
    args = parser.parse_args()

    assert(args.question in ["Q1", "Q2", "Q3", "Q4", "Q5"])

    question_tag = args.question

    #question_tag = "Q1"
    #question_tag = "Q2"
    #question_tag = "Q3"
    #question_tag = "Q4"

    if question_tag in ["Q1", "Q2"]:
        main_binary(args, question_tag)

    if question_tag in ["Q3"]:
        main_multi_method(args, question_tag)

    # TODO" maybe we drop it
    if question_tag in ["Q4"]:
        main_choice(args, question_tag)

    if question_tag in ["Q5"]:
        main_binary_method(args, question_tag)
