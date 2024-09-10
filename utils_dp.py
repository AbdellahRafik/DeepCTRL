
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch

import torch
import torch.nn.functional as F

def calculate_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two tensors.

    Args:
        x (torch.Tensor): The first tensor for correlation calculation.
        y (torch.Tensor): The second tensor for correlation calculation.

    Returns:
        torch.Tensor: The Pearson correlation coefficient between the two tensors.
    """
    # Center the tensors by subtracting the mean
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    
    # Compute the Pearson correlation coefficient
    correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    
    return correlation


def Custom_loss_rule(quantity, price, correlation_threshold=-0.1):
    """
    Custom loss function that penalizes when the correlation between quantity and price 
    is below a given threshold.

    Args:
        quantity (torch.Tensor or np.ndarray): The quantity values to evaluate.
        price (torch.Tensor or np.ndarray): The corresponding price values.
        correlation_threshold (float, optional): The minimum acceptable correlation value 
                                                 between quantity and price. Defaults to -0.1.

    Returns:
        torch.Tensor: The computed custom loss, with a penalty if the correlation 
                      is higher than the specified threshold.
    """
    # Calculate the correlation between quantity and price
    correlation = calculate_correlation(quantity, price)
    
    # Calculate the loss based on unit price differences
    unit_price_diff = price[1:] - price[:-1]
    loss = torch.mean(F.relu(unit_price_diff))  # Penalize positive price differences
    
    # Add a penalty if the correlation is above the threshold
    if correlation > correlation_threshold:
        penalty = torch.abs(correlation - correlation_threshold)
        loss += penalty
    
    return loss


def verification(quantity, price, threshold=-0.1):
    """
    Returns the ratio of samples that follow the rule:
    If quantity increases, the price should decrease.

    Args:
        quantity (torch.Tensor or np.ndarray): The quantity values to verify.
        price (torch.Tensor or np.ndarray): The price values corresponding to the quantity.
        threshold (float, optional): The threshold below which the price is considered to have decreased. Defaults to 0.0.

    Raises:
        ValueError: If the dimensions of `quantity` and `price` do not match.

    Returns:
        float: The ratio of samples where an increase in quantity leads to a decrease in price.
    """
    if isinstance(quantity, torch.Tensor):
        # Ensure quantity and price have the same size
        if quantity.shape != price.shape:
            raise ValueError(f"Dimensions of quantity ({quantity.shape}) and price ({price.shape}) do not match.")
        
        # Calculate differences in price and quantity
        delta_quantity = quantity[1:] - quantity[:-1]
        delta_price = price[1:] - price[:-1]
        # Check that an increase in quantity leads to a decrease in price
        valid_samples = torch.sum((delta_quantity > 0) & (delta_price < threshold))
        return 1.0 * valid_samples / (quantity.shape[0] - 1)
    else:
        # For numpy arrays
        if quantity.shape != price.shape:
            raise ValueError(f"Dimensions of quantity ({quantity.shape}) and price ({price.shape}) do not match.")
        
        delta_quantity = np.diff(quantity)
        delta_price = np.diff(price)
        valid_samples = np.sum((delta_quantity > 0) & (delta_price < threshold))
        return 1.0 * valid_samples / (quantity.shape[0] - 1)