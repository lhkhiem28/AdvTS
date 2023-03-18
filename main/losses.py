import os, sys
from libs import *

def discrepancy(
    augmented_features, features, 
):
    normalized_augmented_features, normalized_features,  = augmented_features/torch.norm(augmented_features, p = 2), features/torch.norm(features, p = 2), 
    discrepancy = torch.dist(
        normalized_augmented_features, normalized_features, 
        p = 2, 
    )

    return discrepancy