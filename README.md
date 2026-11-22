# A3C Model for Atari Kung-Fu Master(Google Colab Edition)

An implementation of the **Asynchronous Advantage Actor-Critic(A3C)** reinforcement learning algorithm trained to play the Atari 2600 game *Kung-Fu Master* using deep reinforcement learning. This project runs completely in the cloud using **Google Colab**,taking advantage of free GPU acceleration.

[![Open In Colab](https://google.com)](https://google.com)

*Click the above badge to open the notebook directly in Google Colab and run the agent immediately!*

---

## Key Features
* **Zero Local Setup:** No need to install complex Atari ROMs, CUDA drivers, or heavy Python dependencies on your laptop. 
*  **Colab Parallelization:** Optimized to handle multi-threaded asynchronous actor-learners within Colab's virtual CPU environments.
*  **Virtual Screen Recording:** Uses 'pyvirtualdisplay' and gymnasium video wrappers to render and save the agent's live gameplay directly inside the cloud browser.

--- 
  
## How to Run the Project

Since this project lives entirely in a Jupyter Notebook('.ipynb'), you don't need to clone it using a terminal.

1. **Click the "Open in Colab" badge** at the top of this page.
2. In the Colab menu, click **Runtime** > **Change runtime type**, and select **T4 GPU**(or any available hardware accelerator).
3. Run the notebook cells sequentially:
   * **Cell 1: Installing Gymnasium**
   * **Cell 2: Importing the libraries**
   * **Cell 3: Creating the architecture of the Neural Network**
   * **Cell 4: Setting up the environment**
   * **Cell 5: Initializing the hyperparameters**
   * **Cell 6: Implementing the A3C class**
   * **Cell 7: Initializing the A3C agent**
   * **Cell 8: Evaluating our A3C agent on a certain number of episodes**
   * **Cell 9: Managing multiple environments simultaneously**
   * **Cell 10: Training the A3C agent**
   * **Cell 11:  Visualizing the results**
  
## Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| 'lr' | 0.0001 | Shared learning rate(Adam) |
| 'gamma' | 0.99 | Discount factor for future rewards |
| 'number_environments' | 10 | Multiple agents trained in multiple environments in parallel |

--- 

## Results & Video Output

*Tip: Running the final cell generates a video file. You can view it directly in colab or download it from the left file sidebar.*

![A3C Agent Playing Kung-Fu Master](D:\AI_Projects\a3c-kungfu\gameplay.gif)

  ---

  ## Model Architecture

  The agent uses **Asynchronous Advantage Actor-Critic(A3C)** architecture built in PyTorch that processes stacked grayscale Atari frames and outputs both action probabilities and state value estimates simultaneously.

  ### 1. Feature Extraction Backbone(Shared CNN)
  The network takes a 4-frame grayscale state buffer as input and passes it through three convolutional layers with **ReLU** activations to extract spatial and motion features:
  * **Input:** 4 stacked grayscale frames
  * **Conv1:** 32 filters, '3x3' kernel, 'stride=2' + ReLU
  * **Conv2:** 32 filters, '3x3' kernel, 'stride=2' + ReLU
  * **Conv3:** 32 filters, '3x3' kernel, 'stride=2' + ReLU
  * **Flatten & Dense Layer:** Flattened into a 512-unit vector and passed through a fully connected layer ('Linear(512, 128)') + ReLU.
  
  ### 2. Dual Actor Critic Heads
  From the shared 128-unit representation, the network splits into two separate output branches:
  * **Actor Head ('fc2a'):** A linear layer ('Linear(128, action_size)') followed by a **Softmax** function to produce a probability distribution over the discrete game actions (Move, Jump, Punch, Kick).
  * **Critic Head ('fc2s'):** A linear layer ('Linear(128, 1)') that outputs a single scalar value estimating the expected return ($V(s)$) of the current state.
  
  ### 3.Optimization & Loss Function
  The network is trained end-to-end using the **Adam** optimizer by minimizing a combined loss objective:
  * **Actor Loss:** Policy gradient loss weighted by the advantage ($A = R + \gamma V(S') - V(s)$), plus an **entropy regularization bonus** ('0.001' weight) to encourage exploration.
  * **Critic Loss:** Mean squared Error ('F.mse_loss') between the predicted state value and the Bellman target value. 

