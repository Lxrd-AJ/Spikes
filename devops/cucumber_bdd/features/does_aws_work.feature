Feature: Does AWS work?
    Everybody wants to know if AWS works

    Scenario Outline: Can i create an EC2 machine
        Given i have created an ec2 machine
        When I tag it with data
        Then I should get that data back