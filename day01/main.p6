

enum Part <A B>;


sub part_a(Str $input) returns Int {
    my $pos = 50;
    my $count = 0;
    for $input.IO.lines -> $line {
        if $line.substr(0, 1) eq 'L' {
            $pos = ($pos - Int($line.substr(1))) % 100;
        } else {
            $pos = ($pos + Int($line.substr(1))) % 100;
        }
        if $pos == 0 {
            $count += 1;
        }
    }
    return $count;
}

sub MAIN(Str $input, Part $part = A) {
    given $part {
        when Part::A {
            my $count = part_a($input);
            say $count;
        }
        when Part::B {
            say "not impl";
        }
        default {
            say "No $part implemented";
        }
    }
}