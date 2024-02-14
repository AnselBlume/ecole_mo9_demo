DEFAULT_SYS_PROMPT = 'You are a helpful assistant.'

LIST_PARTS_PROMPT = '''List the key visible parts of the following object as a bulleted list.
For example, if I said "pogo stick", you might respond:
- handlebars
- shaft
- foot pegs

Example 2: If the object is "laptop", the response could be:
- screen
- keyboard
- touchpad

Example 3: If the object is "dog", the response could be:
- ears
- eyes
- nose
- snout
- tail
- legs
- paws
- body

Provide only the bulleted list without additional commentary. If the object does not have distinct parts (e.g. a ball), repeat the specified object without any bullets (e.g. "ball", not "- ball").

List the key visible parts of the following object: '''

LIST_ATTRIBUTES_PROMPT = '''List the distinctive visual features necessary to identify the following object in a photograph. List these features under two categories: 'Required' for essential attributes and 'Likely' for common but non-essential attributes. For instance, in identifying a school bus, the lists could be:

Required:
- yellow
- black stripes
- wheels
- windows
- bus

Likely:
- school children
- stop sign

Please format your answer following the example provided. If a concept lacks distinctive visual features, clearly state 'none' for that category to indicate the absence of identifiable features. For example, if an object does not have any 'Required' features, you should respond with:

Required:
none

Likely:
- List any common but non-essential attributes here, if applicable.

If there are no 'Likely' attributes either, format your response as:

Likely:
none

You should only provide features that are likely to be visible. For example, saying that a tree frog has "sticky toe pads" is unhelpful for identification, as this feature is not visible in a photograph.
Similarly, you should not provide explanations, justifications, or additional commentary. Instead of saying for a tree frog:

- bright coloration (although not all tree frogs are brightly colored, many species exhibit vibrant hues)

You should just say
- bright coloration

These attributes should be such that they could be a caption for a part of the image. For example, saying for a bus
- school children nearby or inside

would not be a reasonable caption for the part of the image showing the children. Instead, you would just say:
- school children

List the distinctive visual features necessary to identify the following object in a photograph:
'''