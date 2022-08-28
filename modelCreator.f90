module model
    !---------------
    ! adding any number of layers to our neural network
    !----------------


    ! ===============================================================
    ! USE filereader !<loading in weights, biases
    USE activation_functions !<getting activation functions
    USE model_layers
    USE readTester
    ! ===============================================================

    
    IMPLICIT NONE
    contains
    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================

        SUBROUTINE use_model(input, output)
            IMPLICIT NONE
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:,:) :: input
            REAL, ALLOCATABLE, INTENT(OUT), DIMENSION(:,:,:,:) :: output
            REAL :: T1, T2
            CALL CPU_TIME(T1)
            
            !========MaxPool Layer============
            CALL max_pool(input,maxpoolLayers(1), 0,         (/0, 0, 0, 0/),         (/3, 3/))

            call CPU_TIME(T2)

            output = input

        end SUBROUTINE
    !===================================================================================================================================================
        

END module model

!=======================================================================================
