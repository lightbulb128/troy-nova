// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "examples.h"

using namespace std;
using namespace troy;

int main()
{
    while (true)
    {
        cout << "+---------------------------------------------------------+" << endl;
        cout << "| The following examples should be executed while reading |" << endl;
        cout << "| comments in associated files in native/examples/.       |" << endl;
        cout << "+---------------------------------------------------------+" << endl;
        cout << "| Examples                   | Source Files               |" << endl;
        cout << "+----------------------------+----------------------------+" << endl;
        cout << "|  1. BFV Basics             |  1_bfv_basics.cu           |" << endl;
        cout << "|  2. Encoders               |  2_encoders.cu             |" << endl;
        cout << "|  3. Levels                 |  3_levels.cu               |" << endl;
        cout << "|  4. BGV Basics             |  4_bgv_basics.cu           |" << endl;
        cout << "|  5. CKKS Basics            |  5_ckks_basics.cu          |" << endl;
        cout << "|  6. Rotation               |  6_rotation.cu             |" << endl;
        cout << "|  7. Serialization          |  7_serialization.cu        |" << endl;
        cout << "| 10. BFV MatMul             | 10_bfv_matmul.cu           |" << endl;
        cout << "| 11. CKKS MatMul            | 11_ckks_matmul.cu          |" << endl;
        cout << "| 12. LWEs                   | 12_lwes.cu                 |" << endl;
        cout << "| 13. Ring2k                 | 13_ring2k.cu               |" << endl;
        cout << "| 14. BFV Conv2d             | 14_bfv_conv2d.cu           |" << endl;
        cout << "| 15. Batched Operation      | 15_batched_operation.cu    |" << endl;
        cout << "| 20. Memory Pools           | 20_memory_pools.cu         |" << endl;
        cout << "| 30. Issue of Multithread   | 30_issue_multithread.cu    |" << endl;
        cout << "| 99. Quickstart             | 99_quickstart.cu           |" << endl;
        cout << "+----------------------------+----------------------------+" << endl;

        int selection = 0;
        bool valid = true;
        do
        {
            cout << endl << "> Run example by typing number, or 0 to exit: ";
            if (!(cin >> selection))
            {
                valid = false;
            }
            else
            {
                valid = true;
            }
            if (!valid)
            {
                cout << "  [Beep~~] valid option." << endl;
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
        } while (!valid);

        switch (selection)
        {
        case 1:
            example_bfv_basics();
            break;

        case 2:
            example_encoders();
            break;

        case 3:
            example_levels();
            break;

        case 4:
            example_bgv_basics();
            break;

        case 5:
            example_ckks_basics();
            break;

        case 6:
            example_rotation();
            break;

        case 7:
            example_serialization();
            break;

        case 10:
            example_bfv_matmul();
            break;

        case 11:
            example_ckks_matmul();
            break;

        case 12:
            example_lwes();
            break;

        case 13:
            example_ring2k();
            break;

        case 14:
            example_bfv_conv2d();
            break;

        case 15:
            example_batched_operation();
            break;

        case 20:
            example_memory_pools();
            break;

        case 30:
            example_issue_multithread();
            break;

        case 99:
            example_quickstart();
            break;

        case 0:
            troy::utils::MemoryPool::Destroy();
            return 0;
        }
    }

    return 0;
}
