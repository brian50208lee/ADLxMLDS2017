# Conditional Anime GAN

## Environment
python3.6
tensorflow 1.3
scipy (image io)

## Usage
- Train

`python3 main.py --train`

- Test <br/>

`python3 main.py --test --test_text [testing_text.txt]`

or

`bash run.sh [testing_text.txt]`

## Condition
- Hair Color

`['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair']` 

- Eyes Color

`['gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']`

- Hair Style

`['long hair', 'short hair']`