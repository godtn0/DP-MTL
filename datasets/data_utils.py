import numpy as np

import torch


class PadUptoEnd:
    """for pre-transforms, add pad values upto max_seq.
    normally should be called after sampling, like GetOnlyRecent.

    Attributes:
        max_seq (int): maximun_sequence size.
        pad_value_dict (Optional[Dict[str, Any]], optional):
            dict of pad_values in __call__.
            Defaults to None.

    <Example>
    pad_value_dict = {
        "user_id": 0,
        "item_id": 0,
        "elapsed_time_in_ms": 0.0,
        "is_correct": 2,
    }
    pad_upto_end = PadUptoEnd(25, pad_value_dict)

    sample = array([(1804002,  4188, 33000., 0), (1804002, 11239, 27000., 0),
       (1804002,  4121, 28000., 1), (1804002,  8392, 45000., 0),
       (1804002,   606, 19000., 0), (1804002,  5108, 38000., 0),
       (1804002,  8389, 37000., 0), (1804002, 10115, 25000., 0),
       (1804002,  6327, 31000., 0), (1804002,  4476, 38000., 1),
       (1804002,  4529, 39000., 1), (1804002, 17364, 31000., 0),
       (1804002, 19145, 37000., 1), (1804002,  5061, 41000., 1),
       (1804002,  4236, 29000., 0), (1804002,  5738, 41000., 0),
       (1804002, 19353, 29000., 1), (1804002,  4321, 34000., 1),
       (1804002,  5487, 34000., 1)],
      dtype=[
          ('user_id', '<i8'),
          ('item_id', '<i8'),
          ('elapsed_time_in_ms', '<f8'),
          ('is_correct', 'i1')
        ])   # len(sample) = 19

    pad_upto_end(sample)
    >> {
    'elapsed_time_in_ms': array([33000., 27000., 28000., 45000., 19000., 38000., 37000., 25000.,
        31000., 38000., 39000., 31000., 37000., 41000., 29000., 41000.,
        29000., 34000., 34000.,     0.,     0.,     0.,     0.,     0.,
            0.]),
    'is_correct': array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 2, 2, 2,
        2, 2, 2], dtype=int8),
    'item_id': array([ 4188, 11239,  4121,  8392,   606,  5108,  8389, 10115,  6327,
            4476,  4529, 17364, 19145,  5061,  4236,  5738, 19353,  4321,
            5487,     0,     0,     0,     0,     0,     0]),
    'pad_mask': array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False,  True,  True,  True,  True,  True,  True]),
    'sequence_size': array([19]),
    'user_id': array([1804002, 1804002, 1804002, 1804002, 1804002, 1804002, 1804002,
        1804002, 1804002, 1804002, 1804002, 1804002, 1804002, 1804002,
        1804002, 1804002, 1804002, 1804002, 1804002,       0,       0,
                0,       0,       0,       0])
    }
    """

    def __init__(
        self,
        max_seq: int,
        pad_value_dict: Optional[Dict[str, Any]] = None,
    ):
        """Initialization PadUptoEnd with max_seq and pad_value_dict"""
        self.max_seq = max_seq
        self.pad_value_dict = pad_value_dict

    def __call__(self, sample: np.ndarray) -> Dict[str, Any]:
        """Do pad_upto_end
        and convert input into Dict[str, Any] from numpy structured ndarray.
        It will also generate "pad_mask" & "sequence_size"

        Args:
            sample (np.ndarray): single user interactions,
                type of structured_ndarray

        Raises:
            AssertionError: if length of sample is larger than self.max_seq.
                To avoid it, please consider to use other sampling transforms,
                like 'GetOnlyRecent', 'GetRandomSample', 'GetRandomWindow'
                before calling this method.
        """

        assert len(sample) <= self.max_seq
        sample = _pad_upto_end(sample, self.max_seq, pad_value_dict=self.pad_value_dict)
        return sample

def _pad_upto_end(
    sample: Union[np.ndarray, dict],
    max_seq: int,
    pad_value_dict: dict,
    return_type: type = dict,
):
    """pad upto end"""
    if isinstance(sample, np.ndarray) and len(sample.shape) == 1:
        keys = sample.dtype.names
    elif isinstance(sample, dict):
        keys = sample.keys()
    else:
        raise TypeError("please check sample type")

    if return_type == dict:
        res = {}
    else:
        logging.warning("padding into np.structured ndarray can cause bugs")
        raise NotImplementedError()

    seq_size = len(sample)
    res["sequence_size"] = np.array([seq_size], np.int64)
    res["pad_mask"] = np.arange(max_seq) >= seq_size

    for key in keys:
        cur_val = sample[key]
        pad_value = pad_value_dict[key]

        # get pad position value
        if len(cur_val.shape) == 1:
            pad_pos = (0, max_seq - seq_size)
        elif len(cur_val.shape) == 2:  # tags
            pad_pos = ((0, max_seq - seq_size), (0, 0))
        else:
            raise NotImplementedError()

        # append pad
        res[key] = np.pad(
            cur_val,
            pad_pos,
            mode="constant",
            constant_values=pad_value,
        )
    return res

def convert_to_dictofndarray(inputs):
    dict_of_ndarray = {}
    key_list = inputs[0].keys()
    for key in key_list:
        samples = []
        for input_samples in inputs:

            samples.append(input_samples[key].squeeze())

        dict_of_ndarray[key] = np.stack(samples, axis=0)
    return dict_of_ndarray


def standard_collate_fn(interaction_batch):
    interaction_batch = convert_to_dictofndarray(interaction_batch)
    type_dict = {
        "sequence_size": np.int8,
        "pad_mask": bool,
        "item_id": np.int8,
        "is_correct": np.int8,
        "student_choice": np.int8,
        "correct_choice": np.int8,
    }
    for key, interaction_val in interaction_batch.items():
        interaction_val.astype(type_dict[key])
        if key == "pad_mask":
            interaction_val = np.where(interaction_val, False, True)
        interaction_batch[key] = torch.from_numpy(interaction_val)
    return interaction_batch


def enem_collate_fn(batch):
    res_batch = {}
    score_batch = np.stack([x["score"] for x in batch])
    res_batch["score"] = torch.from_numpy(score_batch.astype(np.float32))
    for key in batch[0].keys():
        if "interactions" in key:
            interaction_batch = standard_collate_fn([x[key] for x in batch])
            res_batch[key] = interaction_batch

    return res_batch