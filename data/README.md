## Data formate
create a pre-processed CSV in the below formate:

| id                  | category  | text  |
| -------             | ---       | ---   |
| 1221110283428671488 | _         | Please Reconsider Bill s.288. CDC has announced all EVALI cases were caused by THC products containing Vitamin E Acetate! None were caused by Flavored Nicotine E-Liquid! Adult lives matter too! Devices popular with kids already banned!    |

## PreProcessing

```
pip install tweet-preprocessor
python
>>> import preprocessor as p
>>> input_file_name = "sample_txt.txt"
>>> p.clean_file(file_name, options=[p.OPT.URL, p.OPT.MENTION])
```
