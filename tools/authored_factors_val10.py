# -*- coding: utf-8 -*-
"""Hand-authored factor decomposition of the 100 val10 InstanceNav descriptions.

Each tuple: (attribute, value, source_span) or (relation, reference_category,
reference_attributes, source_span). source_span must be a verbatim substring of
the episode's intrinsic/extrinsic description; tools/build_factor_file.py checks it.
"""

FACTORS = {
0: dict(category="chair", attributes=[
    ("material", "wood and black leather", "made of wood and black leather"),
    ("backrest color", "brown", "Its backrest is brown"),
    ("seat color", "black", "the seat is black"),
    ("frame structure", "two parallel wooden bars joined by a small triangle piece",
     "the frame consists of two parallel wooden bars connected at the ends by a small triangle piece"),
], relations=[], notes="extrinsic 明确说明无法描述周围环境，关系为空"),
1: dict(category="chair", attributes=[
    ("body color", "beige", "two beige chairs"),
    ("cushion color", "white", "white chair cushions"),
    ("frame color", "brownish-yellow (appearance only, texture unclear)", "brownish-yellow in tone"),
], relations=[
    ("next to", "table", ["white", "round"], "placed next to a white round table"),
    ("has on top", "pillow", ["yellow", "small"], "two yellow small pillows on it"),
], notes="描述为复数 two beige chairs，需人工确认指哪一把；右侧电视只是画面构图，不作为关系"),
2: dict(category="plant", attributes=[
    ("material", "metal", "made of metal"),
    ("color", "green, red, yellow and white", "colored green, red, yellow, and white"),
    ("leaf shape", "irregularly shaped leaves", "irregularly shaped leaves"),
], relations=[
    ("in", "flowerpot", ["gray", "large"], "include a gray large flowerpot"),
    ("with", "soil", ["brown"], "brown soil"),
], notes=""),
3: dict(category="plant", attributes=[
    ("body color", "green", "whose body is green"),
    ("leaf surface", "covered with perforated leaves", "covered with perforated leaves"),
    ("leaf edge color", "dark green edges, lighter tips", "The edges of the leaves are dark green, while the tips are lighter in color"),
    ("form", "potted plant", "shows a potted plant"),
], relations=[
    ("in", "pot", [], "potted plant"),
    ("location", "corner of a room", [], "located in the corner of a room"),
    ("next to", "window", [], "next to the window"),
], notes="品系推断（Zygophyllidae）证据不足，未收为因子"),
4: dict(category="toilet", attributes=[
    ("color", "white", "The toilet in this image is white"),
    ("appearance", "clean and fresh", "clean and fresh appearance"),
], relations=[
    ("on top of", "toilet paper", ["two rolls"], "two toilet papers on the toilet"),
    ("next to", "cleaning brush", [], "a cleaning brush next to it"),
    ("on top of", "bottles", [], "several bottles placed on top of the toilet paper rolls"),
], notes=""),
5: dict(category="chair", attributes=[
    ("cover color", "black", "has black covers"),
    ("leg color", "brown", "brown legs"),
], relations=[
    ("on", "floor", ["wooden"], "a wooden floor"),
    ("near", "desk", ["rectangular", "black"], "a rectangular black desk"),
], notes="周围椅子被描述为 blue chairs，与目标的黑色座面不一致，关系仅取整句环境，需人工确认"),
6: dict(category="chair", attributes=[
    ("material pattern", "black and brown striped", "made of black and brown striped material"),
], relations=[
    ("location", "corner of a bedroom", [], "located in the corner of a bedroom"),
    ("near", "bed", [], "There are four pillows on the bed"),
], notes=""),
7: dict(category="bed", attributes=[
    ("frame material", "white wood", "made of white wood"),
    ("blanket color", "white", "has a white blanket"),
    ("pillows", "two pillows on the left side", "two pillows on the left side"),
], relations=[
    ("near", "ladder", [], "a ladder"),
    ("near", "drawers", ["two sets"], "two sets of drawers"),
], notes=""),
8: dict(category="sofa", attributes=[
    ("color", "white", "The sofa in this image is white"),
    ("material", "fabric", "made of fabric"),
], relations=[
    ("on top of", "carpet", ["black and brown patterns"], "placed on top of the carpet, which has black and brown patterns"),
    ("on one side", "pillows", ["five: two gray and white, one blue, one pink, one light blue"], "On one side of the sofa, there are five pillows"),
    ("above", "painting", ["white, gray and black"], "Above the sofa is a painting with white, gray, and black colors"),
], notes=""),
9: dict(category="chair", attributes=[
    ("color", "black", "The chairs in this picture are black"),
    ("backrest shape", "rectangular bend", "their backs have a rectangular bend"),
    ("material", "leather", "the material of the chair is leather"),
], relations=[
    ("near", "table", ["black", "rectangular", "wooden"], "a black rectangular wooden table"),
], notes=""),
10: dict(category="plant", attributes=[
    ("color", "green", "a green potted plant"),
    ("leaf color", "dark green", "dark green leaves"),
    ("form", "potted plant", "a green potted plant"),
], relations=[
    ("location", "center of the room", [], "located in the center of the room"),
    ("on the left side", "curtains", ["beige"], "There are beige curtains and a white door on the left side of the plant"),
    ("on the left side", "door", ["white"], "a white door on the left side of the plant"),
], notes="镜面反射造成的银色外观未收为主要属性"),
11: dict(category="tv_monitor", attributes=[
    ("color", "black", "is black"),
    ("material", "plastic", "appears to be made of plastic"),
    ("condition", "bent and damaged", "which has been bent and damaged"),
    ("screen side", "busted", "a busted side of the TV screen"),
], relations=[
    ("near", "door", ["white", "open"], "an open white door"),
], notes=""),
12: dict(category="chair", attributes=[
    ("material", "wood", "made of wood"),
    ("color", "brown", "its color is brown"),
], relations=[], notes="extrinsic 明确说明画面中没有其它物体"),
13: dict(category="plant", attributes=[
    ("color", "green", "is green"),
    ("leaf shape", "long slender leaves", "many long, slender leaves"),
], relations=[
    ("on", "table", ["white", "square"], "surrounded by a white square table"),
    ("with", "soil", ["brown"], "brown soil on top of the table"),
], notes=""),
14: dict(category="chair", attributes=[
    ("color", "brown", "is brown"),
    ("material", "wood", "made of wood"),
    ("metal parts", "silver", "silver metal parts"),
], relations=[
    ("at", "table", ["long", "white", "rectangular"], "a long white rectangular table"),
], notes="周围另有三把 brownish-yellow office chairs，需确认目标并非其中之一"),
15: dict(category="bed", attributes=[
    ("material", "wooden", "made of wooden material"),
    ("headboard", "wood", "Its headboard, as well as the two sides at both ends of the bed, are also constructed from wood"),
    ("blanket color", "white", "There are white blankets on top"),
    ("pillows", "green pillows underneath", "green pillows underneath"),
], relations=[
    ("on each side", "cabinet", ["brown"], "On each side of the bed, there is a brown cabinet"),
], notes="画面中有两张床，描述提到 covers both beds"),
16: dict(category="plant", attributes=[
    ("color", "brown and green", "The plant in the image is brown and green"),
    ("form", "potted plant", "a potted plant"),
], relations=[
    ("near", "vases", ["two"], "There are two vases and a potted plant in the image"),
], notes=""),
17: dict(category="chair", attributes=[
    ("material", "wooden", "The material of the chair in this photo is wooden"),
    ("color", "brown", "its color is brown"),
], relations=[
    ("on", "carpet", [], "The chair and the carpet in this photo are the only objects visible"),
], notes=""),
18: dict(category="chair", attributes=[
    ("color", "brown", "is brown"),
    ("backrest color", "white", "with a white backrest"),
    ("metal parts", "silver", "silver metal parts"),
], relations=[
    ("at", "table", ["brownish-yellow", "big"], "a brownish-yellow big table"),
], notes=""),
19: dict(category="toilet", attributes=[
    ("color", "white", "The toilet in this picture is white"),
    ("lid", "white lid", "has a white lid"),
], relations=[
    ("in", "bathroom", [], "located in the bathroom"),
    ("to the left", "cabinet", ["green", "white countertop"], "To its left is a green cabinet with a white countertop on top"),
    ("behind", "wall", ["white bricks and tiles"], "behind it are white bricks and tiles"),
    ("to the right", "shower area", ["gray floor and wall"], "On the right side of the toilet is a shower area with a gray floor and wall"),
], notes=""),
20: dict(category="chair", attributes=[
    ("color", "black", "The chair in this image is black"),
    ("material", "leather", "made of leather"),
], relations=[
    ("in front of", "window", [], "in front of a window"),
    ("next to", "wall", [], "next to another wall"),
], notes="intrinsic 说 leather，extrinsic 说 black plastic chair，材料存在冲突，需人工裁决"),
21: dict(category="plant", attributes=[
    ("material", "wood", "made of wood"),
    ("color", "brown", "has a brown color"),
], relations=[
    ("near", "vase", ["black"], "a black vase"),
    ("near", "refrigerator", [], "a refrigerator"),
    ("near", "oven", [], "an oven"),
], notes=""),
22: dict(category="chair", attributes=[
    ("material", "wood", "made of dark brown wooden material"),
    ("color", "dark brown", "made of dark brown wooden material"),
], relations=[
    ("at", "table", ["dark brown", "rectangular"], "a dark brown rectangular table"),
    ("among", "chairs", ["eight", "identical", "wooden"], "there are eight identical wooden chairs"),
], notes=""),
23: dict(category="bed", attributes=[
    ("quilt color", "light green", "made up with a light green quilt"),
    ("quilt pattern", "dark patterns", "which has some dark patterns on it"),
], relations=[
    ("on the right side", "chair", [], "a chair on the right side"),
    ("next to", "table", ["with a lamp"], "a table with a lamp next to the bed"),
    ("in front of", "dresser", [], "a dresser in front of the bed"),
], notes=""),
24: dict(category="tv_monitor", attributes=[
    ("material", "metal", "made of metal"),
    ("color", "black", "appears black"),
    ("condition", "heavily damaged", "that has been heavily damaged"),
    ("orientation", "back facing the camera", "with its back facing the camera"),
], relations=[
    ("in front of", "wall", ["white"], "behind the television is a white wall"),
    ("near", "door", ["white"], "there is a white door in the upper left corner"),
    ("near", "chest of drawers", ["old", "wooden"], "an old wooden chest of drawers"),
], notes=""),
25: dict(category="chair", attributes=[
    ("material", "metal", "made of metal"),
    ("legs", "four legs, two black and two green", "has four legs, two are black and two are green"),
], relations=[
    ("near", "table", [], "There are three chairs and a table around"),
    ("next to", "pet cage", ["pink"], "Next to them is a pink pet cage"),
], notes=""),
26: dict(category="chair", attributes=[
    ("material", "brown leather", "made of brown leather"),
    ("pattern", "black and white patterns", "with black and white patterns on it"),
], relations=[], notes="画面中只有椅子，无周围物体"),
27: dict(category="bed", attributes=[
    ("color", "white (shines under sunlight)", "shines white under sunlight"),
    ("pillows", "two pillows", "with two pillows on the bed"),
], relations=[
    ("on both sides", "bedside tables", ["beige", "two"], "On both sides are two beige bedside tables"),
    ("on the tables", "desk lamps", ["white"], "white desk lamps on them"),
    ("next to", "chest", ["black"], "There is also a black chest next to the bed"),
], notes=""),
28: dict(category="chair", attributes=[
    ("seat surface color", "dark purple", "The chair surface is dark purple"),
    ("frame color", "black", "the frame and armrests are black"),
    ("armrest color", "black", "the frame and armrests are black"),
], relations=[
    ("on", "carpet", [], "The picture only contains a chair and the carpet"),
], notes=""),
29: dict(category="bed", attributes=[
    ("frame material", "brown wood", "made of brown wooden material"),
    ("sheet color", "light blue", "the bed sheets are light blue in color"),
    ("pillow 1", "dark blue cushion with an anchor pattern", "one is a dark blue cushion with an anchor pattern"),
    ("pillow 2", "beige cushion", "the other is a beige cushion"),
    ("small pillow", "square with anchor design", "a small square pillow with a design resembling an anchor"),
], relations=[
    ("on both sides", "windows", ["two"], "There are two windows on both sides of the bed"),
    ("above", "wall", ["green", "with black English words"], "Above the bed, there is a green wall with black English words written on it"),
], notes=""),
30: dict(category="chair", attributes=[
    ("material", "wood", "made of wood"),
    ("color", "brown", "brown in color"),
], relations=[
    ("on the right side", "window", ["white", "white shutters"], "a white window with white shutters on the right side of the chair"),
    ("above", "desk", ["brown", "wooden"], "above the chair is a brown wooden desk"),
    ("near", "plant", ["green"], "a green plant"),
], notes=""),
31: dict(category="chair", attributes=[
    ("colors", "white and black, large white area on top", "mainly made of white and black materials, with a large area of white on top"),
    ("seat edge decoration", "silver nails", "silver nails decorating the edges of the seat cushions"),
], relations=[
    ("at", "table", ["dark brown", "wooden"], "a dark brown wooden table in the middle of the picture"),
    ("near", "bowl", ["silver metal", "decorative", "with green plants and flowers"], "a silver metal decorative bowl filled with green plants and flowers"),
    ("next to", "stool", ["black", "small"], "there is a black small stool"),
], notes=""),
32: dict(category="toilet", attributes=[
    ("color", "white", "The toilet in this picture is white"),
    ("material", "porcelain", "a regular porcelain-made toilet"),
], relations=[
    ("near", "cabinet", ["wooden"], "a wooden cabinet"),
    ("near", "shower stall", ["transparent glass"], "a transparent glass shower stall"),
], notes=""),
33: dict(category="chair", attributes=[
    ("seat color", "red", "with a red seat"),
    ("body", "black padded", "a black padded chair"),
], relations=[
    ("in front of", "desk", ["white"], "placed in front of a white desk"),
    ("behind the desk", "monitors", ["two", "dark"], "There are also two dark monitors behind the desk"),
], notes="与 episode 65 目标描述相同"),
34: dict(category="chair", attributes=[
    ("material", "blue velvet", "made of blue velvet material"),
    ("color", "two contrasting shades of blue", "has two contrasting shades of blue"),
], relations=[
    ("above/beside", "paintings", ["two"], "There are two paintings hanging on the wall above and beside the chair"),
    ("below", "carpet", ["light-colored", "gray stripes"], "a light-colored carpet with gray stripes below the chair"),
    ("on the right side", "round table", ["black", "with beige lamp shade"], "there is a black round table with a beige lamp shade"),
], notes=""),
35: dict(category="bed", attributes=[
    ("size", "double bed", "a double bed"),
    ("comforter color", "white", "filled with white comforter"),
    ("pillowcases", "striped", "striped pillowcases"),
    ("pillows", "two white, one grey, one brown", "two are white, one is grey, and another is brown"),
], relations=[
    ("headboard", "headboard", ["grey"], "The surrounding objects of the bed include a grey headboard"),
    ("on the right side", "table lamp", ["teal"], "a teal table lamp on the right side of the bed"),
    ("below", "chair", ["reddish-brown"], "a reddish-brown chair below the picture"),
], notes=""),
36: dict(category="chair", attributes=[
    ("material", "fabric", "a dark gray fabric"),
    ("color", "dark gray", "a dark gray fabric"),
], relations=[
    ("location", "center of a living room", [], "located in the center of a spacious living room"),
    ("on both sides", "curtains", ["white", "frosted transparent film"], "there are white curtains with frosted transparent film on them"),
    ("behind", "window", ["floor-to-ceiling", "white curtains"], "Behind the chair, there is a floor-to-ceiling window"),
    ("in front of", "coffee table", ["silver", "two white desk lamps", "green plant and white vase"], "a silver coffee table with two white desk lamps placed on it"),
], notes=""),
37: dict(category="plant", attributes=[
    ("color", "green", "a green, thick-stemmed potted plant"),
    ("stem", "thick", "thick-stemmed"),
    ("flowers", "white", "It has white flowers"),
    ("leaf edge color", "dark green at the edges", "leaves that are dark green at the edges"),
], relations=[
    ("in", "flower pot", ["white"], "a white flower pot with green plants inside"),
    ("on", "shelf", ["wooden", "in the corner of the room", "black metal brackets"], "located on the wooden shelf in the corner of the room"),
    ("near", "glass bottles", ["clear", "several different sizes"], "several different sizes of clear glass bottles placed on top of it"),
], notes=""),
38: dict(category="chair", attributes=[
    ("color", "black", "The chair in this picture is black"),
    ("material", "metal", "made of metal"),
], relations=[
    ("at", "countertop", ["black"], "surrounded by a black countertop"),
], notes=""),
39: dict(category="bed", attributes=[
    ("size", "double bed", "a double-size iron bed"),
    ("frame material", "iron", "a double-size iron bed"),
    ("bedsheet color", "gray", "covered with a gray bedsheet"),
    ("pillows", "two light gray pillows", "two light gray pillows"),
    ("headboard", "white with curved patterns at both ends", "The bed frame is white and has curved patterns at both ends of the headboard"),
], relations=[
    ("next to", "cabinet", ["white", "three drawers"], "there is a white cabinet with three drawers"),
    ("on one side", "carpet", ["brown", "colorful flower patterns"], "there is a brown carpet with colorful flower patterns"),
    ("on the left wall", "curtain", ["dark purple"], "On the left wall of the room, there is a dark purple curtain"),
], notes=""),
40: dict(category="chair", attributes=[
    ("color", "black", "The chair in this image is black"),
    ("surface", "smooth and reflective", "the surface of the chair looks very smooth and reflective"),
], relations=[
    ("at", "dining table", ["dark wooden"], "a dark wooden dining table"),
    ("on the table", "decorative items", ["several", "black"], "several black decorative items placed on the surface of the table"),
], notes=""),
41: dict(category="plant", attributes=[
    ("species", "monstera", "a monstera plant"),
    ("leaf color", "dark green", "displaying a dark green color"),
], relations=[
    ("in", "glass bottle", ["transparent", "with water"], "in a glass bottle with water"),
    ("in", "bathroom", [], "In the bathroom, there is a plant placed in a glass bottle"),
    ("on the right side", "towels", ["two", "white", "folded"], "On the right side of the plant, there are two white towels folded together"),
    ("next to", "wash basin", ["silver metal"], "there is also a wash basin made of silver metal"),
    ("above", "mirror", ["rectangular", "golden frame"], "Above the plant, there is a rectangular mirror with a golden frame"),
], notes=""),
42: dict(category="tv_monitor", attributes=[
    ("color", "black", "made of black material"),
    ("surface", "glossy", "has a glossy surface"),
], relations=[
    ("lower left corner", "computer", [], "There is a computer in the lower left corner of the television"),
    ("behind", "wall decorations", ["small"], "some small decorations on the wall behind the television"),
], notes=""),
43: dict(category="chair", attributes=[
    ("color", "brown", "is brown"),
    ("material", "wood", "made of wood"),
], relations=[
    ("at", "dining table", ["wooden"], "surrounded by a wooden dining table"),
    ("near", "refrigerator", ["white"], "a white refrigerator"),
    ("near", "cabinet", ["brown"], "a brown cabinet"),
], notes=""),
44: dict(category="chair", attributes=[
    ("material", "metal", "made of metal"),
    ("color", "silver-white", "has a silver-white color"),
], relations=[
    ("in", "kitchen", [], "surrounded by a kitchen"),
    ("at", "dining table", ["brown", "circular"], "a brown circular dining table"),
    ("next to", "refrigerator", ["black and silver"], "a refrigerator in black and silver colors"),
], notes=""),
45: dict(category="chair", attributes=[
    ("material", "wooden", "The material of the chairs in this image is wooden"),
    ("color", "dark brown", "their colors are dark brown"),
    ("surface shapes", "many unusual shapes", "There are many strange shapes on the chair"),
], relations=[], notes="extrinsic 只有一句异常描述，没有可用的周围物体"),
46: dict(category="chair", attributes=[
    ("color", "white", "The chair in this image is white"),
    ("material", "metal", "made of metal"),
    ("backrest", "white curved lines", "There are white curved lines on the back of the chair"),
    ("frame color", "brownish-black", "there is a brownish-black frame"),
    ("legs", "rolling wheels", "The legs of the chair have rolling wheels"),
], relations=[
    ("near", "walls", ["white"], "white walls around it"),
], notes=""),
47: dict(category="bed", attributes=[
    ("blanket color", "purple", "made up with a purple blanket"),
    ("pillows", "white pillows", "white pillows"),
], relations=[
    ("on the right side", "chair", ["next to the window"], "Chair (right side of the picture), located next to the window"),
    ("near", "window", ["white frame"], "Window with a white frame"),
    ("above", "wall paintings", ["blue, brown, yellow and green"], "Wall paintings on the wall above the bed"),
], notes="intrinsic 说紫毯白枕，extrinsic 说灰毯、四枕（一白三黑），存在冲突，需人工裁决"),
48: dict(category="toilet", attributes=[
    ("color", "white", "The toilet in this image is white"),
], relations=[
    ("next to", "trash can", ["black"], "There is a black trash can next to the toilet"),
    ("on the left side", "wall", ["brown"], "The wall on the left side of the toilet is brown"),
    ("on the right side", "shower door", ["glass", "colorful floor inside"], "there is a glass shower door with a colorful floor inside"),
], notes=""),
49: dict(category="sofa", attributes=[
    ("color", "light brown", "Its overall color is light brown"),
    ("material", "leather/fabric mixture (uncertain)", "possibly a mixture of leather and fabric"),
    ("pillow", "gray-white with patterns", "a gray-white pillow with patterns on it"),
], relations=[
    ("next to", "carpet", [], "There is also a carpet next to the sofa"),
], notes="描述只看到沙发的一部分，材料不确定"),
50: dict(category="chair", attributes=[
    ("material", "wood", "made of wood"),
    ("seat cushion color", "gray", "its seat cushion is gray"),
], relations=[
    ("at", "dining table", ["large", "wooden", "six identical wooden chairs"], "a large wooden dining table"),
], notes=""),
51: dict(category="tv_monitor", attributes=[
    ("color", "black", "The TV monitor in the picture is black"),
    ("stand", "silver", "has a silver stand"),
], relations=[
    ("near", "office chair", ["black"], "a black office chair"),
    ("near", "desk", ["white", "with silver laptop"], "a white desk with a silver laptop on it"),
    ("near", "organizer", ["wall-mounted", "three pockets"], "a wall-mounted organizer with three pockets"),
], notes=""),
52: dict(category="bed", attributes=[
    ("bedsheets", "white", "made up of white bedsheets"),
    ("pillows", "white pillows", "pillows"),
    ("blankets", "white blankets", "blankets"),
], relations=[
    ("above", "wall lamp", ["black frame", "white bulb"], "a wall lamp with a black frame and white bulb"),
    ("near", "chair and table", [], "a chair, a table"),
], notes=""),
53: dict(category="chair", attributes=[
    ("material", "black leather", "black leather"),
    ("back material", "smooth plastic", "their backs are made of smooth plastic"),
], relations=[], notes="画面只有椅子的一部分，无周围物体"),
54: dict(category="chair", attributes=[
    ("material", "wood", "made of wood"),
    ("color", "brown", "made of wood and brown"),
], relations=[
    ("location", "corner of a room", [], "located in the corner of a room"),
    ("behind", "sofa", ["red"], "There is also a red sofa behind the chair"),
    ("on one side", "windows", ["two"], "there are two windows on one side of the wall"),
    ("next to", "carpet", ["blue letters"], "next to it is a carpet with blue letters"),
], notes=""),
55: dict(category="chair", attributes=[
    ("material", "brown leather", "made of brown leather"),
], relations=[], notes="extrinsic 明确说明没有周围信息"),
56: dict(category="tv_monitor", attributes=[
    ("color", "black", "The TV monitor in this image is black"),
    ("frame color", "off-white", "the frame around it is off-white"),
], relations=[
    ("on", "television stand", ["white"], "located on a white television stand"),
    ("next to", "potted plant", ["green", "woven basket"], "one is a green potted plant placed in a woven basket"),
    ("next to", "cactus", ["small", "clay pot"], "a small cactus plant placed in a pot made of clay material"),
], notes=""),
57: dict(category="sofa", attributes=[
    ("color", "cyan blue", "The color of the sofa is cyan blue"),
    ("cushion colors", "gray and black", "with the cushions being gray and black"),
    ("material", "soft fabric (uncertain)", "appears to be a soft and comfortable fabric"),
], relations=[
    ("on the left side", "potted plants", ["two", "green leaves", "black decorative vases"], "there are two potted plants with green leaves and black decorative vases"),
    ("in front of", "carpet", ["gray", "white geometric patterns"], "there is a gray carpet with white geometric patterns on it"),
    ("behind", "window", ["metal frame", "glass"], "The window behind the sofa has a metal frame and glass"),
    ("on the right side", "coffee table", ["transparent", "matte finish"], "there is a transparent coffee table with a matte finish"),
], notes=""),
58: dict(category="tv_monitor", attributes=[
    ("color", "black", "The television monitor in this image is black"),
    ("material", "glass", "made of glass material"),
], relations=[
    ("mounted above", "cabinet", ["white"], "mounted on the wall above a white cabinet"),
    ("on the right side", "window", ["transparent glass"], "there is also a window with transparent glass"),
], notes=""),
59: dict(category="plant", attributes=[
    ("color", "green", "a green, unidentified tropical plant"),
    ("leaf shape", "large oval leaves with cuts and lobes", "It has large, oval-shaped leaves with distinctive cuts and lobes on the edges of the leaves"),
], relations=[
    ("on", "coffee table", ["wooden"], "located on a wooden coffee table"),
    ("on", "carpet", ["white and black diamond patterns"], "a carpet with a combination of white and black diamond patterns"),
    ("on the left side", "sofa chair", ["blue"], "there is a blue sofa chair"),
], notes=""),
60: dict(category="tv_monitor", attributes=[
    ("color", "black", "The television monitor in this image is black"),
    ("material", "metal or similar", "made of metal or a similar material"),
], relations=[
    ("near", "soundbar", ["silver"], "a silver soundbar"),
    ("near", "remote controls", ["two", "white"], "two white remote controls"),
    ("near", "entertainment cabinet", ["red, yellow and black patterns"], "an entertainment cabinet with red, yellow, and black patterns"),
], notes=""),
61: dict(category="plant", attributes=[
    ("color", "green", "a green potted plant"),
    ("form", "potted plant", "a green potted plant"),
], relations=[
    ("in", "flowerpot", ["white"], "placed in a white flowerpot"),
    ("on", "countertop", ["white", "above a cabinet"], "located on a white countertop, which is above a cabinet"),
    ("on both sides", "cabinets", ["white", "with drawers", "silver handles"], "to its left and right are white cabinets with drawers"),
    ("above", "lamp", ["golden"], "there is a golden lamp illuminating the area above the plants"),
], notes=""),
62: dict(category="chair", attributes=[
    ("material", "woven rattan", "made of woven rattan"),
    ("color", "dark brown", "the color is dark brown"),
    ("cushion", "white with beige, gray and yellow stripes", "The white cushion on the chair has a mixture of beige, gray, and yellow stripes"),
], relations=[
    ("on the left side", "carpet", ["off-white"], "On the left side of the chair is an off-white carpet"),
    ("on the left side", "window", [], "There is also a window on the left side of the chair"),
], notes=""),
63: dict(category="plant", attributes=[
    ("color", "green", "The plant in this image is green"),
], relations=[
    ("in", "room", ["white", "yellow wooden floors"], "you can see a white large room with yellow wooden floors"),
    ("near", "sculpture", ["white, gray and brown"], "a sculpture made of white, gray, and brown shapes"),
], notes="描述极简，只有颜色；周围信息多为房间陈设"),
64: dict(category="chair", attributes=[
    ("color", "black", "The chair in this image is black"),
    ("condition", "part of skin peeling off", "with part of its skin peeling off"),
    ("backrest material", "metal", "Its backrest and armrests are made of metal"),
    ("armrest material", "metal", "Its backrest and armrests are made of metal"),
], relations=[], notes="extrinsic 明确说明没有其它物体"),
65: dict(category="chair", attributes=[
    ("seat color", "red", "with a red seat"),
    ("body", "black padded", "a black padded chair"),
], relations=[
    ("in front of", "desk", ["white"], "placed in front of a white desk"),
    ("behind the desk", "monitors", ["two", "dark"], "There are also two dark monitors behind the desk"),
], notes="与 episode 33 目标描述相同（同一实例，不同 episode）"),
66: dict(category="chair", attributes=[
    ("material", "wood", "made of wood"),
    ("condition", "very old, peeling paint", "looks very old"),
    ("back material", "brown wood", "The back of the chair is made of brown wood"),
    ("back markings", "black marks", "some black marks on it"),
    ("sticker", "white sticker on the back", "There is also a white sticker on the back of the chair"),
], relations=[
    ("on both sides", "dining table", ["light gray"], "a light gray dining table"),
    ("in front of left", "glass door", ["open"], "an open glass door that leads to another room"),
], notes=""),
67: dict(category="toilet", attributes=[
    ("color", "white", "The toilet in this picture is white"),
    ("style", "regular", "appears to be a regular style toilet"),
], relations=[
    ("next to", "trash bin", ["black"], "next to it, there is a black trash bin"),
    ("to the right", "cabinet", ["white", "two rolls of toilet paper"], "there is a white cabinet that holds two rolls of toilet paper"),
    ("above", "mirror", [], "Above the toilet, there is a mirror"),
], notes=""),
68: dict(category="plant", attributes=[
    ("color", "black", "The plant in this picture is black"),
    ("condition", "wilted", "looks like a wilted potted plant"),
    ("form", "potted plant", "a wilted potted plant"),
], relations=[], notes="extrinsic 明确说明没有可用周围信息"),
69: dict(category="tv_monitor", attributes=[
    ("material", "black metal", "made of black metal"),
    ("surface", "very shiny", "appears very shiny"),
], relations=[
    ("lower left corner", "books", ["many", "black"], "there are many black books placed in the lower left corner"),
    ("on top of the books", "album", ["golden frame"], "There is also an album with a golden frame placed on top of the books"),
], notes=""),
70: dict(category="bed", attributes=[
    ("blanket color", "red", "made up with a red blanket"),
    ("color scheme", "warm", "the overall color scheme is warm"),
], relations=[
    ("on the bed", "pillows", ["one red, one gray"], "two pillows on the bed, one is red and the other is gray"),
    ("on the right side", "nightstand", ["with a lamp"], "there is a nightstand with a lamp on it"),
    ("in front of", "window", ["with curtains"], "there is a window with curtains hanging outside the window"),
], notes=""),
71: dict(category="chair", attributes=[
    ("material", "wood", "made of wood"),
    ("color", "natural brown", "has a natural brown color"),
], relations=[
    ("at", "dining table", ["wooden"], "surrounded by a wooden dining table and chairs"),
], notes=""),
72: dict(category="bed", attributes=[
    ("material", "wood", "made of wood"),
    ("blanket color", "blue", "covered with a blue blanket"),
    ("pillows", "two blue pillows", "two pillows on both sides of the bed, which are blue"),
    ("blanket detail", "white spots on top", "On top of the blanket, there are some white spots"),
], relations=[
    ("next to", "mirror", [], "Next to the bed, there is a mirror"),
    ("behind", "cabinet", ["brown"], "behind it is a brown cabinet"),
], notes=""),
73: dict(category="plant", attributes=[
    ("color", "green", "a green vase-shaped plants"),
    ("shape", "vase-shaped", "a green vase-shaped plants"),
], relations=[
    ("on the left side", "wall", ["white"], "On the left side of the plant is a white wall"),
    ("on the right side", "wall", ["gray"], "on the right side is another gray wall"),
    ("above", "ceiling fan", ["white"], "an open ceiling with a white fan installed in it"),
], notes=""),
74: dict(category="chair", attributes=[
    ("color", "white", "its color is white"),
], relations=[
    ("near", "glass table", [], "surrounded by four walls and a glass table"),
], notes="材料未提及"),
75: dict(category="plant", attributes=[
    ("color", "green", "The plant in this image is green"),
], relations=[
    ("near", "chairs", ["brown", "wooden"], "There are brown wooden chairs and a white door on both sides of the table"),
], notes="extrinsic 主语是桌子而非植物，关系可信度低，需人工确认"),
76: dict(category="chair", attributes=[
    ("color", "light brown", "its color is light brown"),
    ("material", "white (as stated)", "The chair's material is white"),
], relations=[
    ("near", "chairs", ["many other"], "surrounded by a brown wooden floor and many other chairs"),
], notes="原文把 material 写成 white、color 写成 light brown，语义矛盾，需人工裁决"),
77: dict(category="sofa", attributes=[
    ("material", "light gray fabric", "made of a light gray fabric"),
    ("pillows", "black pillows", "has black pillows"),
], relations=[
    ("faces", "tv", [], "faces the TV"),
    ("on both sides", "black pillows", [], "there are black pillows"),
    ("in front of", "coffee table", ["light wooden surface"], "a coffee table with a light wooden surface"),
    ("behind", "wall", ["white"], "There is also a white wall behind the sofa"),
], notes=""),
78: dict(category="chair", attributes=[
    ("material", "solid wood", "made of solid wood"),
    ("color", "dark brown", "have a dark brown color"),
], relations=[
    ("at", "dining table", ["deep brown"], "There is a deep brown dining table"),
    ("among", "chairs", ["four", "dark reddish-brown"], "four dark reddish-brown chairs"),
], notes=""),
79: dict(category="tv_monitor", attributes=[
    ("color", "black", "The TV monitor is black"),
    ("screen", "dark screen background", "with a dark screen background"),
], relations=[
    ("near", "keyboard", ["black"], "a black keyboard"),
    ("near", "mouse", [], "a mouse"),
    ("near", "phones", ["two"], "two phones"),
], notes=""),
80: dict(category="chair", attributes=[
    ("body color", "black", "The chair's body, seat, and legs are all black"),
    ("seat color", "black", "seat"),
    ("leg color", "black", "legs"),
], relations=[], notes="extrinsic 明确说明无法描述周围环境"),
81: dict(category="chair", attributes=[
    ("material", "wood", "The chair is made of wood"),
    ("color", "white", "made of wood and white"),
], relations=[
    ("next to", "sofa", ["white"], "There is a white sofa and table next to the chair"),
    ("next to", "table", ["white"], "a white sofa and table next to the chair"),
], notes=""),
82: dict(category="chair", attributes=[
    ("material", "wood", "a mixed-colored and wooden chair"),
    ("seat color", "blue", "the seat of the chair is blue"),
    ("seat pattern", "white patterns", "has some white patterns on it"),
    ("backrest", "brown wood", "the backrest and armrests are made of brown wood"),
    ("armrests", "brown wood", "the backrest and armrests are made of brown wood"),
], relations=[
    ("location", "center of a room", [], "located in the center of a room"),
    ("near", "television", [], "you can see a television"),
    ("near", "bamboo basket", [], "a bamboo basket"),
    ("near", "potted plants", ["two"], "two potted plants"),
], notes=""),
83: dict(category="chair", attributes=[
    ("type", "office chair", "a black office chair"),
    ("color", "black", "a black office chair"),
    ("seat cushion color", "red", "with a red cushion on the seat"),
    ("wheels color", "silver-white", "Its wheels are silver-white"),
    ("frame material", "metal", "its frame is made of metal"),
    ("back cover", "transparent plastic sheet", "There is also a transparent plastic sheet covering the back of the chair"),
], relations=[
    ("next to", "computer desk", ["gray"], "next to it is an uncovered gray computer desk"),
    ("on the wall", "shirt", ["black", "short-sleeved"], "a black short-sleeved shirt hanging on the wall"),
], notes=""),
84: dict(category="bed", attributes=[
    ("bedsheet color", "white", "The bed sheet and pillowcases in this image are both white"),
    ("pillowcase color", "white", "pillowcases"),
], relations=[
    ("on the right side", "wicker chair", ["white pillow"], "you can see a wicker chair and a white pillow"),
    ("top left corner", "windows", ["two", "unopened"], "there are two unopened windows"),
], notes=""),
85: dict(category="sofa", attributes=[
    ("color", "red", "The sofa in this image is red"),
    ("material", "soft, likely cotton", "made of soft material, most likely cotton"),
], relations=[
    ("on top", "pillows", ["two", "brown"], "it has two brown pillows on top"),
], notes=""),
86: dict(category="chair", attributes=[
    ("material", "wood", "made of wood"),
    ("color", "black", "made of wood and black"),
], relations=[], notes="extrinsic 明确说明没有其它物体"),
87: dict(category="tv_monitor", attributes=[
    ("material", "metal", "made of metal"),
    ("color", "black", "appears black"),
], relations=[
    ("in front of", "wall", ["white", "slightly dirty"], "include a white wall, which is slightly dirty"),
    ("near", "ventilator", ["iron", "gray and white pattern"], "an iron ventilator with a gray and white pattern on it"),
], notes=""),
88: dict(category="chair", attributes=[
    ("material", "black leather", "The chair's material is black leather"),
    ("back pattern", "yellow patterns", "its back has yellow patterns"),
], relations=[
    ("at", "table", ["long", "wooden"], "surrounded by a long wooden table"),
    ("near", "chairs", ["several", "gray"], "several gray chairs"),
    ("on the walls", "paintings", ["yellow"], "with yellow paintings on them"),
], notes=""),
89: dict(category="chair", attributes=[
    ("material", "dark gray fabric", "The material of the chair in this image is dark gray fabric"),
    ("pattern", "no obvious patterns", "there are no obvious patterns or designs on it"),
], relations=[
    ("on both sides", "sofas", ["two", "dark gray"], "on both sides of the chair are two dark gray sofas"),
    ("behind", "dining table", ["wooden", "oak floor"], "behind the chairs are a wooden dining table and an oak floor"),
    ("on the left side", "kitchen cabinet", ["white", "black handles"], "there is also a white kitchen cabinet with black handles"),
    ("on the right side", "refrigerator and microwave", [], "you can see a refrigerator and a microwave"),
], notes=""),
90: dict(category="sofa", attributes=[
    ("color", "gray", "The sofa in this image is gray"),
    ("seat design", "flat-seated with many buttons", "It has a flat-seated design with many buttons on the surface"),
], relations=[
    ("on", "carpet", ["white and black"], "sitting on a white and black carpet"),
    ("behind", "wall", ["white", "grey bricks on top"], "two white walls with grey bricks on top of them"),
    ("next to", "table", ["yellow", "light brown top"], "there is a yellow table with a light brown top"),
    ("on one side", "doorway", ["open"], "there is an open doorway"),
], notes=""),
91: dict(category="chair", attributes=[
    ("color", "purple", "The chair in this image is purple"),
], relations=[], notes="extrinsic 明确说明没有周围信息；材料未知"),
92: dict(category="chair", attributes=[
    ("seat color", "white", "the seat of the chair is white"),
    ("backrest color", "black", "the backrest and legs are black"),
    ("leg color", "black", "the backrest and legs are black"),
], relations=[
    ("at", "table", ["black", "rectangular"], "a black rectangular table"),
    ("on the table", "vase", ["blue", "with green leaves"], "There is also a blue vase placed on the table with green leaves inside"),
    ("near", "windows", ["two"], "two windows"),
], notes=""),
93: dict(category="tv_monitor", attributes=[
    ("color", "black", "The TV monitor in this picture is black"),
], relations=[
    ("near", "partition", ["silver metal"], "surrounded by a silver metal partition"),
    ("near", "computer parts", ["black", "keyboard and mouse"], "black computer parts (including the keyboard and mouse)"),
], notes=""),
94: dict(category="chair", attributes=[
    ("color", "black", "The chair is black"),
    ("material", "metal", "made of metal material"),
], relations=[
    ("next to", "railing bars", ["black"], "There are black railing bars next to the chair"),
    ("above", "floor", ["yellow", "wooden"], "below the table is a yellow wooden floor"),
], notes=""),
95: dict(category="bed", attributes=[
    ("composition", "two single beds", "made of two single beds"),
    ("color", "white", "They are white"),
    ("bedsheet", "light blue and white striped", "light blue and white striped bed sheets"),
    ("pillows", "green", "green pillows on the beds"),
], relations=[
    ("next to", "bed", ["second bed"], "made of two single beds"),
], notes=""),
96: dict(category="chair", attributes=[
    ("back material", "black metal", "The chair's back, legs, and armrests are made of black metal"),
    ("leg material", "black metal", "legs, and armrests are made of black metal"),
    ("armrest material", "black metal", "armrests are made of black metal"),
    ("seat material", "black plastic", "while the seat is black plastic"),
    ("surface residue", "sticky brown plastic", "with sticky brown plastic"),
], relations=[
    ("on", "floor", ["shiny", "brown wooden"], "there are shiny brown wooden floors around it"),
    ("near", "carpet", ["beige", "white patterns"], "There is also a beige carpet on the floor, with white patterns"),
], notes=""),
97: dict(category="chair", attributes=[
    ("color", "white", "The chair in this picture is white"),
    ("material", "leather (likely)", "the material looks like it may be made of leather"),
    ("pillow", "brown pillow on the chair", "There is a brown pillow on the chair"),
], relations=[
    ("near", "window", ["metal frame"], "There is a white sofa chair near the window"),
    ("next to", "small table", ["black"], "Next to the chair, there is also a black small table"),
], notes=""),
98: dict(category="chair", attributes=[
    ("material", "wood", "made of wood"),
    ("color", "brownish-yellow", "The chair is brownish-yellow"),
    ("backrest color", "khaki", "Its backrest is khaki colored"),
    ("seat color", "white", "while the seat is white"),
    ("surface lines", "gray lines", "There are also some gray lines on the chair's surface"),
], relations=[
    ("next to", "table", ["white"], "There is a white table next to the chair"),
    ("on the left side", "oven", [], "On the left side of the table, there is an oven"),
    ("above", "cabinet", ["white", "black stove hood"], "above it is a white cabinet with a black stove hood"),
], notes=""),
99: dict(category="sofa", attributes=[
    ("material", "blue leather", "made of blue leather"),
], relations=[
    ("location", "center of a living room", [], "located in the center of the living room"),
    ("on the left side", "chairs and table", ["two wooden chairs", "dining table with brown chairs"], "On its left side, there are two wooden chairs and a dining table"),
    ("above", "clock", ["black"], "there is a black clock"),
], notes=""),
}
