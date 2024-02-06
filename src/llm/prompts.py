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

Provide only the bulleted list without additional commentary. If the object does not have distinct parts (e.g. a ball), repeat the specified object without any bullets (e.g. "ball", not "- ball").

List the key visible parts of the following object: '''