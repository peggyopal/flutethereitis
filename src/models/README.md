# Models
This directory is the directory that contains the meat and potatoes of our work.

## process_data.py
There are three main functions that are used:
* `get_bal_train()`
* `get_unbal_train()`
* `get_eval()`

These functions will return a dictionary of all of the cleaned data in the
following format:

```python
{
    'video_id': {
        'labels': [str, ...],
        'audio_embedding': [[float32, float32, ...], ...]
    }
}
```

The `labels` value is a string representation of Google's AudioSet labels, for the purpose of this project we only look at `'flute'` and `'didgeridoo'`.  The `audio_embedding` value is a list of (usually) 10 lists, where each inner list represents a single second as a 128-dimensional feature vector represented in 8-bits.  These features are keyed in a larger dictionary by the `video_id` it corresponds with.

### Future work
Rather than looping through the data every time we want to run our models, extract the desired fields (`video_id`, `label`, and `audio_embedding`) into a `CSV`.  Then we can read from that `CSV` instead.
