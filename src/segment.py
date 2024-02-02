'''
    Functions to split a localized region into parts.

    1. Generate crop around region's bounding box
    2. Potentially remove background from object, setting background to zero
    3. Use fine-grained SAM to segment object into parts
'''