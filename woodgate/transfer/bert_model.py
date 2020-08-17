"""
bert_model.py - This module contains the BertModel class
definition
"""
import os


class BertModel:
    """
    BertModel - The BertModel class encapsulates logic related
    to the specific BERT model used for transfer learning.
    """

    #: The `ALLOWED_L_VALUES` attribute is a constant which
    #: represents the number of stacked encoders in BERT model.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: BERT.
    ALLOWED_L_VALUES = [2, 4, 6, 8, 10, 12, 24]

    #: The `ALLOWED_H_VALUES` attribute is a constant which
    #: represents the number of hidden size of the BERT model.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: BERT.
    ALLOWED_H_VALUES = [128, 256, 512, 768, 1024]

    #: The `ALLOWED_A_VALUES` attribute is a constant which
    #: represents the number of head in the attention layers of
    #: the BERT model.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: BERT.
    ALLOWED_A_VALUES = [2, 4, 8, 12, 16]

    #: The `bert_l_param` attribute represents the number of
    #: stacked encoders used for the BERT model. This attribute
    #: is set via the `BERT_L_PARAM`
    bert_l_param: int = int(
        os.getenv(
            "BERT_L_PARAM",
            "12"
        )
    )
    if bert_l_param not in ALLOWED_L_VALUES:
        raise ValueError(
            "L value not allowed: "
            + "L param must an integer value of 2, 4, "
            + "6, 8, 10, 12, or 24"
        )

    bert_h_param: int = int(
        os.getenv(
            "BERT_H_PARAM",
            "768"
        )
    )
    if bert_h_param == ALLOWED_H_VALUES[0]:
        bert_a_param = 2
    elif bert_h_param == ALLOWED_H_VALUES[1]:
        bert_a_param = 4
    elif bert_h_param == ALLOWED_H_VALUES[2]:
        bert_a_param = 8
    elif bert_h_param == ALLOWED_H_VALUES[3]:
        bert_a_param = 12
    elif bert_h_param == ALLOWED_H_VALUES[4]:
        bert_a_param = 16
    else:
        raise ValueError(
            "H value not allowed: "
            + "H param must an integer value of 128, 256, "
            + "512, 768, or 1024"
        )
