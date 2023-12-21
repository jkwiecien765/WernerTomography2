# WernerTomography2


Najważniejszy program - werner_generator_light.py
Umieściłem tam wszystkie potrzebne funkcje do naszych obliczeń i tylko te dzięki temu nie trzeba już żadnej mojej paczki.
W 9-11 linijkach tego kodu ustawia się parametry do funkcji data_save_iterator. Jeżeli ustawi się je na None, to program poprosi o wpisanie ich na standardowe wejście.
Proponuję ustawić w kodzie N=1000 oraz n=300, a Prefix wpisać ze standardowego wejścia.
Co do błędu z pierwiastkiem, to prawdopodobnie już się nie pojawi, a jeśli się pojawi, to program po prostu przeskoczy do następnej próbki.
Program po każdym wygenerowaniu próbki wypisze "Successfully simulated i of n samples" oraz czas symulacji.

Na każdym komputerze musi znaleźć się folder z programem werner_simulator_light.py, parameters.npy oraz pustym folderem dataJK
