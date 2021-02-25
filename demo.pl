:- use_module('src/dfs_main.pl').

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% World Specification %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Model constants and predicates %%%

constant(Person) :- person(Person).

property(read(Person))           :- person(Person).
property(sleep(Person))          :- person(Person).
property(tease(Person1,Person2)) :- person(Person1), person(Person2).

%%% Constant instantiations %%%

person('jess').
person('nick').
person('winston').
person('zelda').

%%% Hard constraints %%%

% One cannot read and sleep at the same time
constraint(forall(x,neg(and(read(x),sleep(x))))).

% If Winston teases somebody, Jess does not tease the same person
constraint(forall(x,imp(tease('winston',x),neg(tease('jess',x))))).

%%% Probabilistic constraints %%%

% Nick likes sleeping over reading
probability(sleep('nick'),top,0.8).
probability(read('nick'),top,0.4).

% Jess likes reading over sleeping
probability(sleep('jess'),top,0.4).
probability(read('jess'),top,0.8).

% Winston teases Nick more often than (he teases) Jess
probability(tease('winston','nick'),top,0.7).
probability(tease('winston','jess'),top,0.3).

% Otherwise: coin flip
probability(_,top,0.5).

%%%%%%%%%%%%%%%%%
%%% Sentences %%%
%%%%%%%%%%%%%%%%%

% a. Winston teases Nick
sentence(((['winston','teases','nick'],tease('winston','nick')))).

% b. Nick sleeps and Jess reads
sentence((['nick','sleeps','and','jess','reads'],and(sleep('nick'),read('jess')))).

% c. Everyone makes fun of Jess
sentence((['everyone','makes','fun','of','jess'],forall(x,tease(x,'jess')))).

% d. Winston teases Nick or Jess
sentence((['winston','teases','nick','or','jess'],or(tease('winston','nick'),tease('winston','jess')))).

% e. A boy is sleeping
sentence((['a','boy','is','sleeping'],or(sleep('nick'),sleep('winston')))).

%%%%%%%%%%%%%%%%%%%
%%% Application %%%
%%%%%%%%%%%%%%%%%%%

%%% Atomic propositions %%%

atomic_props :-
        foreach(property(Prop),(write(Prop),nl)).

%%% Sampling %%%

sample_models :-
        dfs_sample_models(100,Ms),dfs_models_to_matrix(Ms,MM),dfs_pprint_matrix(MM),dfs_write_matrix(MM,'model.observations').

%%% Probability testing %%%

test_probabilities :-
        dfs_read_matrix('model.observations',MM),
        % One cannot read and sleep at the same time
        foreach(person(Person),
                dfs_conj_probability(sleep(Person),read(Person),MM,0)),
        % Nick likes sleeping over reading
        dfs_prior_probability(sleep('nick'),MM,Pr1),
        dfs_prior_probability(read('nick'),MM,Pr2),
        Pr1 > Pr2,
        % Jess likes reading over sleeping
        dfs_prior_probability(sleep('jess'),MM,Pr3),
        dfs_prior_probability(read('jess'),MM,Pr4),
        Pr3 < Pr4,
        % Winston teases Nick more often than (he teases) Jess
        dfs_prior_probability(tease('winston','nick'),MM,Pr5),
        dfs_prior_probability(tease('winston','jess'),MM,Pr6),
        Pr5 > Pr6,
        % If Winston teases somebody, Jess does not tease the same person
        foreach(person(Person),
                dfs_conj_probability(tease('jess',Person),tease('winston',Person),MM,0)).

%%% Sentences %%%

sentence_vectors :-
        dfs_read_matrix('model.observations',MM),
        dfs_sentences(Ss),
        foreach(( member((Sen,Sem),Ss),
                  dfs_vector(Sem,MM,Vec) ),
                ( write(Sen), nl,
                  write(Sem), nl,
                  write(Vec), nl ) ).

%%% Comprehension Score %%%

compute_comprehension :-
        dfs_read_matrix('model.observations',MM),
        dfs_inference_score(sleep('nick'),or(sleep('nick'),sleep('winston')),MM,Cs),
        write(Cs).

%%% Write to file %%%
