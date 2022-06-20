module model_layers

    USE activation_functions
    USE derived_types

    implicit none

contains
    !=== for Gemm operations ======
    subroutine linear_layer(inp, lin, transInp)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: transInp
        real, ALLOCATABLE, intent(inout) :: inp(:,:) !===input is 2d usually (k,n), where n is usually 1
        TYPE(linLayer), INTENT(IN) :: lin !===stores the weights (m,k) and biases (m,1)
        real, DIMENSION(size(lin%weights,1),size(inp,transInp+1)) :: bias_broadcast !==(m,n)
        bias_broadcast = SPREAD(lin%biases, 2, size(inp,transInp+1))
        if (transInp == 0) THEN
            inp = lin%fn_ptr(matmul(inp,TRANSPOSE(lin%weights)) + TRANSPOSE(bias_broadcast))
        ELSE
            inp = lin%fn_ptr(matmul(lin%weights,inp) + bias_broadcast)
        END IF
    end subroutine

    subroutine lstm_cell(input, hid1, cell1, Whh, Wih, Bih, Bhh)
        implicit none
        real, intent(in), DIMENSION(:,:) :: input !== (n,batch_size)
        real, intent(inout), ALLOCATABLE, DIMENSION(:,:) :: hid1 !==(m,batch_size)
        real, intent(inout), ALLOCATABLE, DIMENSION(:,:) :: cell1 !==(m,batch_size)
        real, intent(in), DIMENSION(:,:) :: Whh !==(4m,m)
        real, intent(in), DIMENSION(:,:) :: Wih !==(4m,n)
        real, intent(in), DIMENSION(:) :: Bhh !==(4m,1)
        real, intent(in), DIMENSION(:) :: Bih !==(4m,1)
        real, ALLOCATABLE, DIMENSION(:,:) :: hiddenOut !==(m,batch_size)
        real, ALLOCATABLE, DIMENSION(:,:) :: cellOut


        real, DIMENSION(size(Bhh, 1), size(input,2)) :: Bhh_broadcast !== (4m, batch_size)
        real, DIMENSION(size(Bhh, 1), size(input,2)) :: Bih_broadcast
        real, DIMENSION(size(Whh, dim=1), size(input,2)) :: gates_out
        real, DIMENSION(size(hid1,1),4,size(input,2)) :: chunks !4, size(Whh, dim=1)/4
        ALLOCATE(hiddenOut(size(hid1),size(input,2)))
        ALLOCATE(cellOut(size(cell1),size(input,2)))

        !======= applying bhh and bih to each column of input =====
        Bhh_broadcast = SPREAD(Bhh, 2, size(input,2))
        Bih_broadcast = SPREAD(Bih, 2, size(input,2))

        gates_out = MATMUL(Wih, input) + Bih_broadcast + MATMUL(Whh, hid1) + Bhh_broadcast !==(4m,batch_size) BROADCAST BIAS

        chunks = RESHAPE(gates_out, (/size(gates_out,1)/4, 4, size(input,2)/))

        chunks(:, 1, :) = sigmoid2d(chunks(:, 1, :))
        chunks(:, 2, :) = sigmoid2d(chunks(:, 2, :))
        chunks(:, 3, :) = tanhh2d(chunks(:, 3, :))
        chunks(:, 4, :) = sigmoid2d(chunks(:, 4, :))

        cellOut = (chunks(:, 2, :) * cell1) + (chunks(:, 1, :) * chunks(:, 3, :))
        hiddenOut = chunks(:, 4, :) * tanhh2d(cellOut)
        hid1 = hiddenOut
        cell1 = cellOut
    end subroutine

    subroutine lstm(input, hid1, cell1, Whh, Wih, Bih, Bhh, output)
        implicit none
        real, intent(inout), ALLOCATABLE, DIMENSION(:,:,:) :: input !== (timesteps,batch_size,n)
        real, intent(inout), ALLOCATABLE, DIMENSION(:,:,:) :: hid1 !==(num_directions,batch_size,m), add another dim for num dir
        real, intent(inout), ALLOCATABLE, DIMENSION(:,:,:) :: cell1 !==(num_directions,batch_size,m), add another dim for num dir
        real, intent(in), ALLOCATABLE, DIMENSION(:,:,:) :: Whh !==(num_directions,4m,m), add another dim for num dir
        real, intent(in), ALLOCATABLE, DIMENSION(:,:,:) :: Wih !==(num_directions,4m,n), add another dim for num dir
        real, intent(in), ALLOCATABLE, DIMENSION(:) :: Bhh !==(4m,1)
        real, intent(in), ALLOCATABLE, DIMENSION(:) :: Bih !==(4m,1)
        real, INTENT(OUT), ALLOCATABLE, DIMENSION(:,:,:,:) :: output !==(timesteps,num_directions,m,batch_size)
        ! INTEGER, INTENT(IN) :: nlayers, NEED TO ADD NLAYERS FUNCTIONALITY (no need for now since onnx apparently doesn't support it)
        INTEGER :: timesteps
        INTEGER :: i
        real, ALLOCATABLE, DIMENSION(:,:) :: hid1changed
        real, ALLOCATABLE, DIMENSION(:,:) :: cell1changed
        timesteps = size(input,1)
        input = reshape(input, (/SIZE(input,dim=1),SIZE(input,dim=3), SIZE(input,dim=2)/), order = [1,3,2])
        hid1 = reshape(hid1, (/SIZE(hid1,dim=1),SIZE(hid1,dim=3), SIZE(hid1,dim=2)/), order = [1,3,2])
        cell1 = reshape(cell1, (/SIZE(cell1,dim=1),SIZE(cell1,dim=3), SIZE(cell1,dim=2)/), order = [1,3,2])
        ALLOCATE(output(timesteps,size(hid1,dim=1),size(hid1,dim=2),size(hid1,dim=3)))
        hid1changed = hid1(1,:,:)
        cell1changed = cell1(1,:,:)
        DO i=1, timesteps
            CALL lstm_cell(input(i,:,:), hid1changed, cell1changed, Whh(1,:,:), Wih(1,:,:), Bih, Bhh)
            hid1(1,:,:) = hid1changed
            cell1(1,:,:) = cell1changed
            output(i,:,:,:) = hid1
        END DO
        hid1 = reshape(hid1, (/SIZE(hid1,dim=1),SIZE(hid1,dim=3), SIZE(hid1,dim=2)/), order = [1,3,2]) !==reshaped to (num_directions,batch_size,m)
        cell1 = reshape(cell1, (/SIZE(cell1,dim=1),SIZE(cell1,dim=3), SIZE(cell1,dim=2)/), order = [1,3,2]) !==(num_directions,batch_size,m)
        output = reshape(output,(/SIZE(output,dim=1),SIZE(output,dim=2), SIZE(output,dim=4),SIZE(output,dim=3)/), order = [1,2,4,3]) !==(timesteps,num_directions,batch_size,m)
    end subroutine

    subroutine conv(inp, convWeights, bias, dilations, pads, strides)
        implicit none
        REAL, INTENT(INOUT), ALLOCATABLE, DIMENSION(:,:,:,:) :: inp !==(numImages,imageD1,imageD2)
        REAL, INTENT(IN), ALLOCATABLE, DIMENSION(:,:,:,:) :: convWeights !==(numConvRows,numConvCols,ConvRowDim,ConvColDim)
        REAL, INTENT(IN), ALLOCATABLE, DIMENSION(:) :: bias
        INTEGER, INTENT(IN), DIMENSION(:) :: dilations
        INTEGER, INTENT(IN), DIMENSION(:) :: pads
        INTEGER, INTENT(IN), DIMENSION(:) :: strides
        INTEGER :: in_channels !==numImages SHOULD BE INTENT(IN)
        INTEGER :: out_channels !==numConvCols SHOULD BE INTENT(IN)
        INTEGER :: kernel_size !==(ConvRowDim,ConvColDim) SHOULD BE INTENT(IN)
        REAL, ALLOCATABLE, DIMENSION(:,:,:,:) :: out

        INTEGER :: outer
        INTEGER :: overImage
        INTEGER :: inner
        INTEGER :: outRowDim
        INTEGER :: outColDim
        REAL :: sumini = 0
        in_channels = SIZE(inp, dim=2)
        out_channels = SIZE(convWeights, dim=2)
        kernel_size = SIZE(convWeights, dim=3)
        ALLOCATE(out(1,out_channels, size(inp,dim=3)-kernel_size + 1, size(inp,dim=3)-kernel_size+1))
        outRowDim = size(inp,dim=3)-kernel_size + 1
        outColDim = size(inp,dim=3)-kernel_size + 1

        DO outer = 0, out_channels-1 !==iterating through each output image
            DO overImage = 0, (outRowDim*outColDim)-1 !==iterating kernel through the whole image
                DO inner = 0, in_channels-1 !==applying kernel to each input image
                    sumini = sumini + SUM(inp(1,inner+1,(overImage/outRowDim + 1):(overImage/outRowDim+kernel_size) &
                                      ,(MODULO(overImage,outColDim) + 1):(MODULO(overImage,outColDim)+kernel_size)) &
                          * convWeights(outer+1,inner+1,:,:)) !==depending on the way convWeights is laid out change position of outer/inner, currently it is row based (input images are applied to 1st row then 2nd row, etc.)
                END DO
                out(1,outer+1,overImage/outRowDim + 1,MODULO(overImage,outColDim)+1) = sumini + bias(outer+1)
                sumini = 0;
            END DO
        END DO
        inp = out
        DEALLOCATE(out)

    END subroutine


    subroutine max_pool(inp, maxpool, ceil_mode, pads, strides) !==ceil_mode, pads, strides
        implicit none
        REAL, INTENT(INOUT), ALLOCATABLE, DIMENSION(:,:,:) :: inp !==(numImages,imageD1,imageD2)
        TYPE(maxpoolLayer), INTENT(IN) :: maxpool !==(ConvRowDim,ConvColDim)
        INTEGER, INTENT(IN) :: ceil_mode
        INTEGER, INTENT(IN), DIMENSION(:) :: pads
        INTEGER, INTENT(IN), DIMENSION(:) :: strides
        !==INTEGER, INTENT(IN) :: stride IMPLEMENT THIS
        REAL, ALLOCATABLE, DIMENSION(:,:,:) :: out
        INTEGER :: kernel_size

        INTEGER :: overImage
        INTEGER :: inner
        INTEGER :: outRowDim
        INTEGER :: outColDim
        kernel_size = maxpool%kernel_size
        ALLOCATE(out(size(inp,dim=1), size(inp,dim=2)/kernel_size, size(inp,dim=3)/kernel_size))
        outRowDim = size(inp,dim=2)/kernel_size
        outColDim = size(inp,dim=3)/kernel_size

        DO overImage = 0, (outRowDim*outColDim)-1 !==iterating kernel through the whole image
            DO inner = 0, size(inp,dim=1)-1 !==applying kernel to each input image
                out(inner+1,(overImage/outRowDim)+1,MODULO(overImage,outColDim)+1) = &
                 MAXVAL(inp(inner+1,((overImage/outRowDim)*kernel_size + 1):((overImage/outRowDim)*kernel_size)+kernel_size &
                 ,(MODULO(overImage,outColDim)*kernel_size + 1): &
                (MODULO(overImage,outColDim)*kernel_size+kernel_size)))
            END DO
        END DO
        inp = out
        DEALLOCATE(out)

    end subroutine



end module model_layers
