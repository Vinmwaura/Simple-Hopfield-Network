# Simple-Hopfield-Network
Implementation of a simple discrete Hopfield network.

A Hopfield network is a form of recurrent artificial neural network where each unit is connected to every other unit. The net has symmetric weights with no self-connections i.e w_ij = w_ji and w_ii=0.

The network serves as content-addressable ("Associative") memory systems with binary threshold nodes which means they can store patterns into their weight matrix. When given an input that contains some part of a pattern stored in memory (Cue) the network will update each node in the network asynchronously or synchronously until it converges and restores the association/pattern that was stored in the weight matrix i.e reconstructs the patterns it has memorised.

**Updating a Node**

To update a node when given an input vector **V**, we first calculate the total input to node:

V_in = &#x2211; (W_ji * V_j)

If V_in >= threshold value(e.g 0), Set activation for node V_i to 1

Otherwise set activation for node V_i to 0


Iterate randomly for all the nodes until no change occurs anymore i.e no node in the network changes values during an iterations. 

**Training**

To store patterns into a network use the following formula:


W_ij = &#x2211; (2 * V_i - 1)(2 * V_j - 1) where you sum over all patterns presented.



References
* https://en.wikipedia.org/wiki/Hopfield_network
* http://staff.itee.uq.edu.au/janetw/cmc/chapters/Hopfield/
* http://web.cs.ucla.edu/~rosen/161/notes/hopfield.html
