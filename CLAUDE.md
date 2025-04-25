Always run things via uv run $file

Always add packages via uv add $dep

Import things via absolute imports


Design code so you can run things and validate as you go (without requiring too much time to run each file, always add debug mode that runs quickly), and break into as small changes as possible and test them incrementally

Use git! after each change, make git commits etc

IMPORTANT: You must always break changes into the smallest unit possible -- if I ask you to do something, NEVER just go off and start editing files unless it's trivial. Instead, propose a plan, ask for my input, and then make the smallest increment change possible and run it every time to confirm it works

Prioritize simple code that works on one path -- if there's a bug with data shape, find the source of the bug rather than adding defnesive checks. Similarly, when refactoring unless I explicitly ask don't create legacy versions of the code to support both -- just update it so everything works with the new one.