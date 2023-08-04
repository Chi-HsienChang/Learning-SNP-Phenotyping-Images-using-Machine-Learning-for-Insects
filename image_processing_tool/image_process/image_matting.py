from rembg import remove

input_path = './TEST.JPG'
output_path = './TEST_MASK.JPG'

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)


