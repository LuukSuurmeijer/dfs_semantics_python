# DFS Semantics Python

This repo is essentially a python implementation (incomplete) of https://github.com/hbrouwer/dfs-tools , notably without sampling. Logic is handled by NLTK's logic package. 

The goal of this repo is to implement a traditional lambda-calculus style compositional system for Distributional Formal Semantics, i.e. vectors of truth values. This results in the ability to compositionally build sentences from elementary words and phrases. Crucially the compositional trees are graded and each reduction adjusts fuzzy inferential prediction with respect to the Vector Space we are working in.

An example of such a derivation in 3 dimensions:

