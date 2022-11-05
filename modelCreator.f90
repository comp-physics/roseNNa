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
            REAL (c_double), INTENT(INOUT), DIMENSION(        1, 1, 28, 28) :: i0
            !===========================

            !INPUTS CORRESPONDING TO INTERMEDIARY PROCESSING
            REAL (c_double), ALLOCATABLE,  DIMENSION(:,:) :: output0
            REAL (c_double), ALLOCATABLE,  DIMENSION(:,:) :: output1
            !================================================

            !OUTPUTS CORRESPONDING TO C
            REAL (c_double), INTENT(OUT), DIMENSION(        1, 10) :: o0
            !===========================

            REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: Input3

            REAL :: T1, T2

            ALLOCATE(Input3(        1, 1, 28, 28))

            Input3 = i0

            
            CALL CPU_TIME(T1)
            
            !========Reshape============
            reshapeLayers(1)%reshape4d = RESHAPE(reshapeLayers(1)%reshape4d,(/SIZE(reshapeLayers(1)%reshape4d, dim = 4),&
                & SIZE(reshapeLayers(1)%reshape4d, dim = 3), SIZE(reshapeLayers(1)%reshape4d, dim = 2),&
                & SIZE(reshapeLayers(1)%reshape4d, dim = 1)/), order = [4, 3, 2, 1])
            output0 = RESHAPE(reshapeLayers(1)%reshape4d,(/256, 10/), order = [2, 1])

            !========Conv Layer============
            CALL conv(Input3, convLayers(1)%weights, convLayers(1)%biases,         (/1, 1/),         (/2, 2, 2, 2/),         (/1,&
                & 1/))
            !===========Add============
            Input3 = Input3 + RESHAPE(broadc(addLayers(1)%adder,        (/1, 8, 28, 28/),RESHAPE(        (/3, 28, 4, 28/),       &
                & (/2, 2/), order=[2,1])),         (/1, 8, 28, 28/))

            Input3 = relu4d(Input3)
            
            !========MaxPool Layer============
            CALL max_pool(Input3,maxpoolLayers(1), 0,         (/0, 0, 0, 0/),         (/2, 2/))

            !========Conv Layer============
            CALL conv(Input3, convLayers(2)%weights, convLayers(2)%biases,         (/1, 1/),         (/2, 2, 2, 2/),         (/1,&
                & 1/))
            !===========Add============
            Input3 = Input3 + RESHAPE(broadc(addLayers(2)%adder,        (/1, 16, 14, 14/),RESHAPE(        (/3, 14, 4, 14/),       &
                & (/2, 2/), order=[2,1])),         (/1, 16, 14, 14/))

            Input3 = relu4d(Input3)
            
            !========MaxPool Layer============
            CALL max_pool(Input3,maxpoolLayers(2), 0,         (/0, 0, 0, 0/),         (/3, 3/))

            !========Reshape============
            Input3 = RESHAPE(Input3,(/SIZE(Input3, dim = 4), SIZE(Input3, dim = 3), SIZE(Input3, dim = 2), SIZE(Input3, dim =&
                & 1)/), order = [4, 3, 2, 1])
            output1 = RESHAPE(Input3,(/1, 256/), order = [2, 1])

            !=======MatMul=========
            CALL matmul2D(output1, output0)
            
            !===========Add============
            output1 = output1 + RESHAPE(addLayers(3)%adder,         (/1, 10/))

            call CPU_TIME(T2)
            o0 = output1
        end SUBROUTINE
    !===================================================================================================================================================
        

END module model

!=======================================================================================
