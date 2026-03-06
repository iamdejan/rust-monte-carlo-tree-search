# Code Generation

Hi Kilo Code agent! Whatever model you are using, make sure you adhere to the guide on this file to generate the code.

## Rules of engagement

1. You're allowed to create functions, structs and impl blocks to generate the code.
2. With the exception of import statements and Cargo macros (allow and deny), explain the flow of the program using comments. Make sure that you write not only what the program is doing, but why. It will help me to judge your work.
3. You have to add Rustdoc. The Rustdoc should contain what is the function for (basically the description), a brief summary of the steps, and input and output parameters. If your function has the ability to throw panic, please state it in the Rustdoc as well.
4. You're not allowed to add `#[allow(dead_code)]`.
