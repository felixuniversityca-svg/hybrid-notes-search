# The TCP three-way handshake

Before two machines exchange data over TCP they synchronise with a three-way
handshake. The client sends a SYN segment, the server replies with SYN-ACK, and
the client answers with an ACK. After those three messages both sides agree on
the starting sequence numbers and the connection is established. The same pattern
in reverse, using FIN and ACK segments, tears the connection down cleanly so no
data is lost in flight.
