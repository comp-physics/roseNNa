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
        
        SUBROUTINE use_model(        i0,         o0)
            IMPLICIT NONE
            !INPUTS CORRESPONDING TO C
            REAL (c_double), INTENT(INOUT), DIMENSION(        1, 2, 6, 6) :: i0
            !===========================

            !INPUTS CORRESPONDING TO INTERMEDIARY PROCESSING
            !================================================

            !OUTPUTS CORRESPONDING TO C
            REAL (c_double), INTENT(OUT), DIMENSION(        1, 2, 2, 2) :: o0
            !===========================

            REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: input

            REAL :: T1, T2

            ALLOCATE(input(        1, 2, 6, 6))

            input = i0

            
            CALL CPU_TIME(T1)
            
            !========MaxPool Layer============
            CALL avgpool(input,avgpoolLayers(1), 0,         (/0, 0, 0, 0/),         (/3, 3/))

            call CPU_TIME(T2)
            o0 = input
        end SUBROUTINE
    !===================================================================================================================================================
        

END module model

!=======================================================================================
